#pragma once
#include <stdint.h>

// ==== Signal parameters ====
constexpr int   WIN_SAMPLES = 128;
constexpr int   IMU_AXES    = 3;
constexpr float FS_HZ       = 2750.0f;
constexpr float F0_HZ       = 95.0f;
constexpr float DELTA_HZ    = 25.0f;


// ==== Feature parameters ====
constexpr int   FEAT_DIM    = 7;


// Feature Order: ["jerk_rms", "centroid_norm", "zcr", "crest_factor_amp", "ber_f0", "high_ratio", "bandpower_over_entropy_amp"]

// Mean values for the 7 features
const float FEATURE_MEANS[] = {2574.62630955f, 0.40435631f, 0.29729900f, 1.21566447f, 0.30328685f, 6.65608516f, 5171.23649665f};

// Scale values (std dev) for the 7 features
const float FEATURE_SCALES[] = {1328.35332233f, 0.04485641f, 0.09398709f, 0.09826379f, 0.22738502f, 3.09645829f, 14776.61296602f};


// Function to compute the 7 features for model
void compute_features(const float imu[WIN_SAMPLES][IMU_AXES], float out_features[FEAT_DIM]);

// Apply z-score standardization using the hardcoded values
inline void standardize_features(float features[FEAT_DIM]) {
  for (int i = 0; i < FEAT_DIM; i++) {
    
    if (FEATURE_SCALES[i] != 0.0f) {
      features[i] = (features[i] - FEATURE_MEANS[i]) / FEATURE_SCALES[i];
    }
  }
}