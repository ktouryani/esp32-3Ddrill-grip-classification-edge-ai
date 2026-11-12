#include <Arduino.h>
#include <Adafruit_LSM6DSOX.h>
#include <SPI.h>

#include "features.h"      
#include "labels.h"        
#include "model_int8.h"    

#include "esp_timer.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"


tflite::MicroErrorReporter micro_error_reporter;

// --- Runtime config ---
static constexpr int   kStride  = 64;
static constexpr int   kBufSize = 256;
static const int kNumClasses = 5;

// --- SPI / IMU pins ---
#define SPI_CS_PIN   5
#define SPI_SCK_PIN  18
#define SPI_MISO_PIN 19
#define SPI_MOSI_PIN 23
SPIClass spi(VSPI);
Adafruit_LSM6DSOX imu;

// --- ERM (Vibration Motor) ---
#define ERM_PIN   13
#define ERM_CH    0
#define ERM_FREQ  5000
#define ERM_BITS  8
static void erm_setup() { ledcSetup(ERM_CH, ERM_FREQ, ERM_BITS); ledcAttachPin(ERM_PIN, ERM_CH); ledcWrite(ERM_CH, 0); }
static inline void erm_set(uint8_t duty) { ledcWrite(ERM_CH, duty); }

// --- TFLite ---
constexpr int kArenaSize = 30 * 1024;
uint8_t tensor_arena[kArenaSize];
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor *in_imu = nullptr, *in_feat = nullptr, *out_cls = nullptr;
struct QP { float scale; int zp; };
static QP q_in_imu, q_in_feat, q_out_cls;

// --- Data Buffers ---
static float    rb[kBufSize][IMU_AXES];
static int      rb_head  = 0;
static int      rb_count = 0;
static int      samples_since_last_infer = 0;
static float win_buf[WIN_SAMPLES][IMU_AXES];

// --- Majority Vote Globals ---
static const int kVoteK = 11; // How many past predictions to average over
static int vote_buf[kVoteK];
static int vote_counts[kNumClasses] = {0};
static int vote_idx = 0, vote_n = 0;

// --- PERF / Telemetry ---
static uint32_t perf_infer_count = 0;
static uint64_t perf_infer_us_sum = 0;
static uint64_t perf_feat_us_sum  = 0;
static uint32_t perf_started_ms = 0;
static float    perf_preds_per_sec = 0.0f;

// --- Run Gating ---
static bool     g_runArmed = false;
static bool     g_inferEnabled = false;
static uint32_t g_inferEnableAtMs = 0;
static const uint32_t kStartDelayMs = 1500;

// ================== Tiny helpers ==================
static inline int8_t quantize(float x, float s, int zp) {
  int32_t q = (int32_t)roundf(x/s + zp);
  return (int8_t)constrain(q, -128, 127);
}
static inline float dequantize(int8_t q, float s, int zp) { return (float)(q - zp) * s; }

// Majority Vote Function ---
static int majority_push(int cls){
  if (vote_n == kVoteK) { int old = vote_buf[vote_idx]; vote_counts[old]--; }
  else { vote_n++; }
  vote_buf[vote_idx] = cls; vote_counts[cls]++;
  vote_idx = (vote_idx + 1) % kVoteK;
  int best=0, bc=vote_counts[0];
  for (int c=1; c<kNumClasses; c++){ if (vote_counts[c] > bc){ bc = vote_counts[c]; best = c; } }
  return best;
}

// --- Ring buffer helpers ---
static inline void rb_push(float ax, float ay, float az) {
  rb[rb_head][0] = ax; rb[rb_head][1] = ay; rb[rb_head][2] = az;
  rb_head = (rb_head + 1) % kBufSize;
  if (rb_count < kBufSize) rb_count++;
}
static void rb_extract_lastN(int N, float tmp[][IMU_AXES]) {
  int start = (rb_head - N + kBufSize) % kBufSize;
  for (int i=0; i<N; i++){
    int j = (start + i) % kBufSize;
    tmp[i][0] = rb[j][0]; tmp[i][1] = rb[j][1]; tmp[i][2] = rb[j][2];
  }
}

// ================== Serial commands ==================
static void print_perf_summary() {
  if (perf_infer_count == 0) return;
  uint32_t elapsed_ms = millis() - perf_started_ms;
  float secs = elapsed_ms / 1000.0f;
  float avg_infer_ms = (perf_infer_us_sum / (float)perf_infer_count) / 1000.0f;
  float avg_feat_ms  = (perf_feat_us_sum  / (float)perf_infer_count) / 1000.0f;
  perf_preds_per_sec = (secs > 0) ? perf_infer_count / secs : 0.0f;
  Serial.println("----- PERF SUMMARY -----");
  Serial.printf("Predictions: %lu  | Elapsed: %.2f s | Preds/sec: %.2f\n",
                (unsigned long)perf_infer_count, secs, perf_preds_per_sec);
  Serial.printf("Avg inference: %.2f ms | Avg features: %.2f ms\n", avg_infer_ms, avg_feat_ms);
  Serial.println("------------------------");
}

static void handle_serial_commands() {
  if (!Serial.available()) return;
  String line = Serial.readStringUntil('\n'); line.trim();
  if (!line.length()) return;
  if (line.startsWith("erm ")) {
    int duty = constrain(line.substring(4).toInt(), 0, 255);
    erm_set((uint8_t)duty); Serial.printf("[ERM] duty=%d\n", duty);
  } else if (line.equalsIgnoreCase("off")) {
    erm_set(0); Serial.println("[ERM] off");
  } else if (line.equalsIgnoreCase("start")) {
    g_runArmed = true; g_inferEnabled = false; g_inferEnableAtMs = millis() + kStartDelayMs;
    Serial.println("[RUN] start: inference will enable after grace delay");
  } else if (line.equalsIgnoreCase("stop")) {
    bool was_running = g_runArmed && g_inferEnabled;
    g_runArmed = false; g_inferEnabled = false; erm_set(0);
    Serial.println("[RUN] stop: inference disabled");
    if (was_running) print_perf_summary();
  } else {
    Serial.println("Commands: erm <0-255> | off | start | stop");
  }
}

// ================== setup ==================
void setup() {
  Serial.begin(921600); delay(2000);
  erm_setup();
  Serial.println("Commands: erm <0-255> | off | start | stop");
  spi.begin(SPI_SCK_PIN, SPI_MISO_PIN, SPI_MOSI_PIN, SPI_CS_PIN);
  if (!imu.begin_SPI(SPI_CS_PIN, &spi)) { Serial.println("LSM6DSOX SPI init FAILED"); while (1); }
  imu.setAccelRange(LSM6DS_ACCEL_RANGE_4_G);
  imu.setAccelDataRate(LSM6DS_RATE_3_33K_HZ);
  imu.setGyroDataRate(LSM6DS_RATE_SHUTDOWN);
  const tflite::Model* model = tflite::GetModel(model_int8_tflite);
  static tflite::MicroMutableOpResolver<7> resolver;
  if (resolver.AddExpandDims()        != kTfLiteOk ||
      resolver.AddConv2D()            != kTfLiteOk ||
      resolver.AddReshape()           != kTfLiteOk ||
      resolver.AddMaxPool2D()         != kTfLiteOk ||
      resolver.AddFullyConnected()    != kTfLiteOk ||
      resolver.AddConcatenation()     != kTfLiteOk ||
      resolver.AddSoftmax()           != kTfLiteOk) {
    Serial.println("Op resolver setup failed");
    while (1) { delay(100); }
  }
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kArenaSize, &micro_error_reporter);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) { Serial.println("AllocateTensors failed. Increase kArenaSize."); while (1); }
  in_imu = interpreter->input(0); in_feat = interpreter->input(1);
  if (in_feat->dims->data[1] != FEAT_DIM) { in_imu = interpreter->input(1); in_feat = interpreter->input(0); }
  if (in_imu->dims->data[1] != WIN_SAMPLES || in_feat->dims->data[1] != FEAT_DIM) { Serial.println("[ERR] Model input shapes mismatch."); while(1); }
  q_in_imu.scale = in_imu->params.scale; q_in_imu.zp = in_imu->params.zero_point;
  q_in_feat.scale = in_feat->params.scale; q_in_feat.zp = in_feat->params.zero_point;
  out_cls = interpreter->output(0);
  q_out_cls.scale = out_cls->params.scale; q_out_cls.zp = out_cls->params.zero_point;
  Serial.println("Setup complete. Type 'start' to begin.");
}

// ================== loop ==================
void loop() {
  handle_serial_commands();
  if (!g_runArmed) { delay(5); return; }
  if (!g_inferEnabled) {
    if (millis() >= g_inferEnableAtMs) {
      g_inferEnabled = true; Serial.println("[RUN] inference enabled");
      perf_infer_count = 0; perf_infer_us_sum = 0; perf_feat_us_sum = 0; perf_started_ms = millis();
      rb_count = 0; rb_head = 0; samples_since_last_infer = 0;
      vote_idx = vote_n = 0; memset(vote_counts, 0, sizeof(vote_counts));
    } else { delay(5); return; }
  }

  sensors_event_t accel, g, t;
  if (imu.getEvent(&accel, &g, &t)) {
    rb_push(accel.acceleration.x, accel.acceleration.y, accel.acceleration.z);
    if (rb_count >= WIN_SAMPLES) samples_since_last_infer++;
  }

  if (rb_count >= WIN_SAMPLES && samples_since_last_infer >= kStride) {
    samples_since_last_infer = 0;
    uint32_t loop_start_us = micros();
    rb_extract_lastN(WIN_SAMPLES, win_buf);
    uint32_t feat_t0 = micros();
    float features[FEAT_DIM];
    compute_features(win_buf, features);
    standardize_features(features);
    uint32_t feat_us = micros() - feat_t0;
    perf_feat_us_sum += feat_us;
    int8_t* imu_q = in_imu->data.int8;
    for (int i=0; i<WIN_SAMPLES; i++) { for (int j=0; j<IMU_AXES; j++) { imu_q[i*IMU_AXES + j] = quantize(win_buf[i][j], q_in_imu.scale, q_in_imu.zp); } }
    int8_t* feat_q = in_feat->data.int8;
    for (int k=0; k<FEAT_DIM; k++) { feat_q[k] = quantize(features[k], q_in_feat.scale, q_in_feat.zp); }
    uint32_t inf_t0 = micros();
    interpreter->Invoke();
    uint32_t inf_us = micros() - inf_t0;
    perf_infer_us_sum += inf_us; perf_infer_count++;
    float scores[kNumClasses];
    int raw_class = 0; float highest_score = -1000.0f;
    for (int i=0; i<kNumClasses; i++) {
      scores[i] = dequantize(out_cls->data.int8[i], q_out_cls.scale, q_out_cls.zp);
      if (scores[i] > highest_score) { highest_score = scores[i]; raw_class = i; }
    }
    
    // --- MAJORITY VOTE SMOOTHING ---
    int final_class = majority_push(raw_class);
    
    float total_ms = (micros() - loop_start_us) / 1000.0f;
    if (perf_infer_count > 1) { perf_preds_per_sec = (perf_preds_per_sec * (perf_infer_count - 1) + (1000.0f / total_ms)) / perf_infer_count; }
    else { perf_preds_per_sec = (1000.0f / total_ms); }

    Serial.printf("Pred: %-25s | inf: %.2f ms | feat: %.2f ms | preds/sec: %.2f\n",
                  LABELS[final_class], (float)inf_us/1000.0f, (float)feat_us/1000.0f, perf_preds_per_sec);
  }
}
