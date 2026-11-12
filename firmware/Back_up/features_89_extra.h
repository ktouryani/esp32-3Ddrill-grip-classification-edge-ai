#pragma once
#include <stdint.h>

// ==== Signal parameters (must match training) ====
// Keep 2750.0f if your trained scalers are for 2.75 kHz.
// If you retrain at 3330 Hz, change this to 3330.0f and set USE_RESAMPLE=0.
constexpr int   WIN_SAMPLES = 128;
constexpr int   IMU_AXES    = 3;
constexpr float FS_HZ       = 2750.0f;   // <- your trained FS
constexpr float F0_HZ       = 95.0f;
constexpr float DELTA_HZ    = 25.0f;

// ==== Feature parameters for the 89% accuracy model ====
constexpr int   FEAT_DIM    = 7;


// Feature Order: ["jerk_rms", "centroid_norm", "zcr", "crest_factor_amp", "ber_f0", "high_ratio", "bandpower_over_entropy_amp"]
// Means:
const float FEATURE_MEANS[] = {2574.62630955f, 0.40435631f, 0.29729900f, 1.21566447f, 0.30328685f, 6.65608516f, 5171.23649665f};
// Scales:
const float FEATURE_SCALES[] = {1328.35332233f, 0.04485641f, 0.09398709f, 0.09826379f, 0.22738502f, 3.09645829f, 14776.61296602f};

constexpr int FEATURE_MEANS_LEN  = (int)(sizeof(FEATURE_MEANS)  / sizeof(FEATURE_MEANS[0]));
constexpr int FEATURE_SCALES_LEN = (int)(sizeof(FEATURE_SCALES) / sizeof(FEATURE_SCALES[0]));

void compute_features(const float imu[WIN_SAMPLES][IMU_AXES], float out_features[FEAT_DIM]);
inline void standardize_features(float features[FEAT_DIM]) {
  const int N_means  = (int)(sizeof(FEATURE_MEANS)/sizeof(FEATURE_MEANS[0]));
  const int N_scales = (int)(sizeof(FEATURE_SCALES)/sizeof(FEATURE_SCALES[0]));
  const int N = (FEAT_DIM < N_means ? FEAT_DIM : N_means);
  const int M = (FEAT_DIM < N_scales ? FEAT_DIM : N_scales);
  const int K = (N < M ? N : M);
  for (int i = 0; i < K; ++i) {
    float s = FEATURE_SCALES[i];
    features[i] = (s != 0.0f) ? (features[i] - FEATURE_MEANS[i]) / s : features[i];
  }
}

