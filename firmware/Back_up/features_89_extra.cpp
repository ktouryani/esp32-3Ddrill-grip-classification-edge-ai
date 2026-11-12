#include "features.h"
#include <cmath>
#include "arduinoFFT.h"

// If you had a features_heap_init() elsewhere, make it a no-op now.
extern "C" void features_heap_init() {}

static inline float array_mean(const float* arr, int len) {
  float s = 0.0f;
  for (int i = 0; i < len; ++i) s += arr[i];
  return s / len;
}

void compute_features(const float imu[WIN_SAMPLES][IMU_AXES],
                      float out_features[FEAT_DIM]) {
  // ---- Magnitude (orientation agnostic) ----
  float mag[WIN_SAMPLES];
  for (int i = 0; i < WIN_SAMPLES; ++i) {
    const float x = imu[i][0], y = imu[i][1], z = imu[i][2];
    mag[i] = sqrtf(x*x + y*y + z*z);
  }

  // 1) Jerk RMS
  float jerk_sum_sq = 0.0f;
  for (int i = 1; i < WIN_SAMPLES; ++i) {
    const float jerk = (mag[i] - mag[i-1]) * FS_HZ;
    jerk_sum_sq += jerk * jerk;
  }
  const float jerk_rms = sqrtf(jerk_sum_sq / WIN_SAMPLES);

  // 2) Zero-crossing rate around mean
  const float mag_mean = array_mean(mag, WIN_SAMPLES);
  int zc = 0;
  for (int i = 1; i < WIN_SAMPLES; ++i) {
    const float a = mag[i-1] - mag_mean;
    const float b = mag[i]   - mag_mean;
    if (a * b < 0) zc++;
  }
  const float zcr = (float)zc / (WIN_SAMPLES - 1);

  // 3) Crest factor (amplitude)
  float ss = 0.0f, vmax = 0.0f;
  for (int i = 0; i < WIN_SAMPLES; ++i) {
    const float v = mag[i];
    ss += v*v;
    if (v > vmax) vmax = v;
  }
  const float mag_rms = sqrtf(ss / WIN_SAMPLES);
  const float crest_factor_amp = vmax / (mag_rms + 1e-9f);

  // ---- Spectral features (Hann + FFT) ----
  // Use local arrays to avoid uninitialized globals and heap issues.
  float vReal[WIN_SAMPLES];
  float vImag[WIN_SAMPLES];

  for (int i = 0; i < WIN_SAMPLES; ++i) {
    vReal[i] = mag[i] - mag_mean;
    vImag[i] = 0.0f;
  }

  // Construct FFT with buffers (portable across ArduinoFFT versions).
  ArduinoFFT<float> FFT(vReal, vImag, WIN_SAMPLES, FS_HZ);
  FFT.windowing(FFT_WIN_TYP_HANN, FFT_FORWARD);
  FFT.compute(FFT_FORWARD);
  FFT.complexToMagnitude();    // vReal now holds |X[k]|

  const int   K  = WIN_SAMPLES / 2;
  const float df = FS_HZ / WIN_SAMPLES;

  // 4) Spectral centroid (normalized to Nyquist)
  float cnum = 0.0f, cden = 0.0f;
  for (int k = 1; k < K; ++k) {
    const float f = k * df;
    const float a = vReal[k];
    cnum += f * a;
    cden += a;
  }
  const float centroid_norm =
      (cden > 1e-9f) ? (cnum / cden) / (FS_HZ * 0.5f) : 0.0f;

  // Total power
  float total_power = 0.0f;
  for (int k = 1; k < K; ++k) {
    const float a = vReal[k];
    total_power += a * a;
  }

  // 5) Band Energy Ratio around F0 ±10 Hz
  int k_lo = (int)((F0_HZ - 10.0f) / df);
  int k_hi = (int)((F0_HZ + 10.0f) / df);
  if (k_lo < 1) k_lo = 1; if (k_hi >= K) k_hi = K - 1;
  float band_power = 0.0f;
  for (int k = k_lo; k <= k_hi; ++k) {
    const float a = vReal[k];
    band_power += a * a;
  }
  const float ber_f0 = (total_power > 1e-9f) ? band_power / total_power : 0.0f;

  // 6) High ratio: power >= 2.5*F0 over power in [F0, 3*F0]
  int k_f0_1   = (int)(F0_HZ / df);
  int k_f0_2_5 = (int)((2.5f * F0_HZ) / df);
  int k_f0_3   = (int)((3.0f  * F0_HZ) / df);
  if (k_f0_1   < 1) k_f0_1   = 1;
  if (k_f0_2_5 >= K) k_f0_2_5 = K - 1;
  if (k_f0_3   >= K) k_f0_3   = K - 1;

  float high_p = 0.0f, band_hr = 0.0f;
  for (int k = 1; k < K; ++k) {
    const float p = vReal[k] * vReal[k];
    if (k >= k_f0_2_5)           high_p  += p;
    if (k >= k_f0_1 && k <= k_f0_3) band_hr += p;
  }
  const float high_ratio = (band_hr > 1e-9f) ? high_p / band_hr : 0.0f;

  // 7) Bandpower over spectral entropy
  double ent = 0.0;
  for (int k = 1; k < K; ++k) {
    const double p = (vReal[k] * vReal[k]) / (total_power + 1e-12);
    if (p > 0) ent -= p * log(p + 1e-12);
  }
  const float spectral_entropy = (float)(ent / log((double)K));

  int k_lo_ent = (int)((F0_HZ - 2 * DELTA_HZ) / df);
  int k_hi_ent = (int)((F0_HZ + 2 * DELTA_HZ) / df);
  if (k_lo_ent < 1) k_lo_ent = 1; if (k_hi_ent >= K) k_hi_ent = K - 1;

  float band_ent = 0.0f;
  for (int k = k_lo_ent; k <= k_hi_ent; ++k) {
    const float a = vReal[k];
    band_ent += a * a;
  }

  const float bandpower_over_entropy_amp =
      (spectral_entropy > 1e-9f)
        ? (band_ent / (total_power + 1e-12f)) / spectral_entropy
        : 0.0f;

  // ---- Output (order must match training) ----
  // ["jerk_rms", "centroid_norm", "zcr", "crest_factor_amp",
  //  "ber_f0", "high_ratio", "bandpower_over_entropy_amp"]
  out_features[0] = jerk_rms;
  out_features[1] = centroid_norm;
  out_features[2] = zcr;
  out_features[3] = crest_factor_amp;
  out_features[4] = ber_f0;
  out_features[5] = high_ratio;
  out_features[6] = bandpower_over_entropy_amp;
}
