#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <string>
#include <filesystem>
#include <cctype>

#include "features.h"
#include "model_int8.h"  // const unsigned char model_int8[];  (your uploaded .h)

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace fs = std::filesystem;

// ---- CSV reader (same as your fixed one) ---- (copy the latest read_csv from the logger)
static std::vector<std::array<float,3>> read_csv(const std::string& path);
/* paste your latest read_csv implementation here (header-aware: ax,ay,az) */

// ---- util ----
static inline int8_t quantize(float x, float scale, int zp){
  int v = (int)std::round(x/scale + zp);
  if (v < -128) v = -128; if (v > 127) v = 127;
  return (int8_t)v;
}

int main() {
  // 1) Collect files
  const std::string ROOT = "../data/new_test_dataset";
  std::vector<std::string> paths;
  for (auto& d : fs::directory_iterator(ROOT)) {
    if (!d.is_directory()) continue;
    for (auto& f : fs::directory_iterator(d.path())) {
      if (f.is_regular_file() && f.path().extension()==".csv")
        paths.push_back(f.path().string());
    }
  }
  if (paths.empty()){ std::cerr<<"No CSVs\n"; return 1; }

  // 2) TFLM setup
  tflite::MicroErrorReporter micro_reporter;
  tflite::ErrorReporter* error_reporter = &micro_reporter;
  const tflite::Model* model = tflite::GetModel(model_int8);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    std::cerr<<"Schema mismatch\n"; return 1;
  }

  tflite::AllOpsResolver resolver;
  constexpr int kArenaSize = 1200*1024; // adjust if needed
  static uint8_t tensor_arena[kArenaSize];
  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kArenaSize, error_reporter);
  if (interpreter.AllocateTensors() != kTfLiteOk) { std::cerr<<"AllocateTensors failed\n"; return 1; }

  TfLiteTensor* in0 = interpreter.input(0); // imu
  TfLiteTensor* in1 = interpreter.input(1); // feats
  TfLiteTensor* out = interpreter.output(0);

  float imu_scale = in0->params.scale; int imu_zp = in0->params.zero_point;
  float feat_scale= in1->params.scale; int feat_zp= in1->params.zero_point;
  float out_scale = out->params.scale; int out_zp = out->params.zero_point;

  std::ofstream flog("../logs/predictions_cpp.csv");
  flog << "file,start,pred,probs\n";

  // 3) iterate files/windows
  for (auto& p : paths) {
    auto data = read_csv(p);
    if (data.size() < WIN_SAMPLES) continue;

    for (int s=0; s+WIN_SAMPLES<= (int)data.size(); s+=64) {
      // build window
      float imu_win[WIN_SAMPLES][IMU_AXES];
      for (int i=0;i<WIN_SAMPLES;i++)
        for (int j=0;j<IMU_AXES;j++)
          imu_win[i][j] = data[s+i][j];

      // features (raw -> standardize)
      float feats[FEAT_DIM];
      compute_features(imu_win, feats);
      standardize_features(feats);

      // 4) quantize & feed
      // reshape assumptions: (1,128,3) and (1,7)
      int8_t* in0_data = in0->data.int8;
      int idx=0;
      for (int i=0;i<WIN_SAMPLES;i++)
        for (int j=0;j<IMU_AXES;j++)
          in0_data[idx++] = quantize(imu_win[i][j], imu_scale, imu_zp);

      int8_t* in1_data = in1->data.int8;
      for (int k=0;k<FEAT_DIM;k++)
        in1_data[k] = quantize(feats[k], feat_scale, feat_zp);

      // 5) invoke
      if (interpreter.Invoke() != kTfLiteOk) { std::cerr<<"Invoke failed\n"; continue; }

      // 6) read softmax (int8 or float)
      std::vector<float> probs(out->dims->data[out->dims->size-1], 0.f);
      if (out->type == kTfLiteInt8) {
        for (int i=0;i<(int)probs.size();++i) {
          int8_t q = out->data.int8[i];
          probs[i] = (q - out_zp) * out_scale;
        }
      } else if (out->type == kTfLiteFloat32) {
        for (int i=0;i<(int)probs.size();++i) probs[i] = out->data.f[i];
      } else {
        std::cerr<<"Unexpected output type\n"; continue;
      }

      // argmax
      int best=0; float bestv=probs[0];
      for (int i=1;i<(int)probs.size();++i) if (probs[i]>bestv){best=i;bestv=probs[i];}

      // 7) log
      flog << fs::path(p).filename().string() << "," << s << "," << best << ",";
      for (int i=0;i<(int)probs.size();++i) {
        flog << probs[i]; if (i<(int)probs.size()-1) flog << ";";
      }
      flog << "\n";
    }
  }

  std::cout<<"✅ Wrote logs/predictions_cpp.csv\n";
  return 0;
}
