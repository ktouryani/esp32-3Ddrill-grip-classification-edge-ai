#include "features.h"
#include <cmath>
#include "arduinoFFT.h"

// ----- helpers -----
static inline float mean(const float* x, int n) {
  float s = 0.f; for (int i=0;i<n;i++) s += x[i]; return s / n;
}
static inline int freq_to_bin_floor(float freq, float df, int max_bin) {
  if (freq <= 0.f) return 0;
  int idx = static_cast<int>(freq / df);
  return (idx > max_bin) ? max_bin : idx;
}

static inline int freq_to_bin_ceil(float freq, float df, int max_bin) {
  if (freq <= 0.f) return 0;
  float ratio = freq / df;
  int idx = static_cast<int>(ratio);
  if (static_cast<float>(idx) < ratio) idx++;
  if (idx > max_bin) idx = max_bin;
  return idx;
}

// Compute rFFT magnitude into mag[0..K], where K = N/2 (inclusive like numpy rfft)
static void rfft_magnitude_rect(const float* x_in, float* mag_out, int N) {
  // Rectangular window (no windowing) to match Python
  // ArduinoFFT uses in-place buffers; we copy then use its API without windowing.
  static ArduinoFFT<float> FFT;
  static float vReal[WIN_SAMPLES];
  static float vImag[WIN_SAMPLES];

  for (int i=0;i<N;i++){ vReal[i]=x_in[i]; vImag[i]=0.f; }

  FFT.compute(vReal, vImag, N, FFT_FORWARD);
  FFT.complexToMagnitude(vReal, vImag, N);

  int K = N/2;
  for (int k=0;k<=K;k++) mag_out[k] = vReal[k];
}

static void spectral_stats_from_mag(const float* mag, int K, float df,
                                    int k_lo_band, int k_hi_band, int k_lo_high,
                                    float& centroid_norm, float& high_ratio) {
  float num = 0.f;
  float den = 0.f;
  for (int k = 1; k <= K; ++k) { // skip DC like Python's max(S[1:])
    float fk = k * df;
    float val = mag[k];
    num += fk * val;
    den += val;
  }
  float centroid = (den > 1e-9f) ? (num / den) : 0.f;
  centroid_norm = centroid / (FS_HZ * 0.5f);

  float sum_band = 0.f;
  float sum_high = 0.f;
  for (int k = 1; k <= K; ++k) {
    float val = mag[k];
    if (k >= k_lo_high) sum_high += val;
    if (k >= k_lo_band && k <= k_hi_band) sum_band += val;
  }
  high_ratio = (sum_band > 1e-9f) ? (sum_high / sum_band) : 0.f;
}

static float band_energy_ratio_from_power(const float* power, int K,
                                          int k_lo, int k_hi, float total_power) {
  if (k_hi < k_lo) return 0.f;
  float band = 0.f;
  for (int k = k_lo; k <= k_hi && k <= K; ++k) {
    band += power[k];
  }
  return (total_power > 1e-12f) ? (band / total_power) : 0.f;
}

static float spectral_entropy_from_power(const float* power, int K, float total_power) {
  if (total_power <= 1e-12f) return 0.f;
  float H = 0.f;
  const float eps = 1e-12f;
  for (int k = 0; k <= K; ++k) {
    float q = power[k] / total_power;
    if (q > 0.f) H -= q * logf(q + eps);
  }
  float norm = logf(static_cast<float>(K + 1));
  return (norm > 1e-12f) ? (H / norm) : 0.f;
}

void compute_features(const float imu[WIN_SAMPLES][IMU_AXES], float out_features[FEAT_DIM]) {
  static float a[WIN_SAMPLES];
  float sumsq_amp = 0.f;
  float max_amp = 0.f;
  float jerk_sum = 0.f;
  float prev_amp = 0.f;
  for (int i=0;i<WIN_SAMPLES;i++){
    float x = imu[i][0];
    float y = imu[i][1];
    float z = imu[i][2];
    float amp = sqrtf(x*x + y*y + z*z);
    a[i] = amp;
    sumsq_amp += amp * amp;
    if (amp > max_amp) max_amp = amp;
    float j = (i == 0) ? 0.f : (amp - prev_amp) * FS_HZ;
    jerk_sum += j * j;
    prev_amp = amp;
  }
  float jerk_rms = sqrtf(jerk_sum / WIN_SAMPLES);

  // zcr
  float m = mean(a, WIN_SAMPLES);
  int zc=0;
  for (int i=1;i<WIN_SAMPLES;i++){
    float x0=a[i-1]-m, x1=a[i]-m;
    if (x0==0) x0 = 1e-12f;
    if (x1==0) x1 = -1e-12f;
    if (x0*x1 < 0) zc++;
  }
  float zcr = (float)zc / (WIN_SAMPLES-1);

  // Prepare zero-mean amplitude for spectral features
  static float a_dc[WIN_SAMPLES];
  for (int i=0;i<WIN_SAMPLES;i++){
    a_dc[i] = a[i] - m;
  }

  // Compute FFT magnitude once and reuse
  static float mag[WIN_SAMPLES/2 + 1];
  rfft_magnitude_rect(a_dc, mag, WIN_SAMPLES);

  static float power[WIN_SAMPLES/2 + 1];
  float total_power = 0.f;
  const int K = WIN_SAMPLES / 2;
  for (int k=0;k<=K;k++){
    float p = mag[k] * mag[k];
    power[k] = p;
    total_power += p;
  }
  if (total_power < 1e-12f) total_power = 1e-12f;

  const float df = FS_HZ / WIN_SAMPLES;

  // spectral stats
  int k_lo_band = freq_to_bin_ceil(F0_HZ, df, K);
  int k_hi_band = freq_to_bin_floor(3.f * F0_HZ, df, K);
  if (k_hi_band < k_lo_band) k_hi_band = k_lo_band;
  int k_lo_high = freq_to_bin_ceil(2.5f * F0_HZ, df, K);
  float cen_n=0.f, high_ratio=0.f;
  spectral_stats_from_mag(mag, K, df, k_lo_band, k_hi_band, k_lo_high, cen_n, high_ratio);

  // ber_f0 using power
  int ber_lo = freq_to_bin_ceil(F0_HZ - 10.f, df, K);
  int ber_hi = freq_to_bin_floor(F0_HZ + 10.f, df, K);
  float ber_f0 = band_energy_ratio_from_power(power, K, ber_lo, ber_hi, total_power);

  // bandpower_over_entropy_amp like Python (absolute band power / entropy)
  int bpoe_lo = freq_to_bin_ceil(F0_HZ - 2.f*DELTA_HZ, df, K);
  int bpoe_hi = freq_to_bin_floor(F0_HZ + 2.f*DELTA_HZ, df, K);
  float ber_band = band_energy_ratio_from_power(power, K, bpoe_lo, bpoe_hi, total_power);
  float Hn = spectral_entropy_from_power(power, K, total_power);
  float band_pow = ber_band * total_power;
  float bpoe = (Hn>1e-12f) ? (band_pow / Hn) : 0.f;

  // crest factor on amplitude
  float rms = sqrtf(sumsq_amp / WIN_SAMPLES);
  float cf = max_amp / (rms + 1e-12f);

  // order: ["jerk_rms","centroid_norm","zcr","crest_factor_amp","ber_f0","high_ratio","bandpower_over_entropy_amp"]
  out_features[0]=jerk_rms;
  out_features[1]=cen_n;
  out_features[2]=zcr;
  out_features[3]=cf;
  out_features[4]=ber_f0;
  out_features[5]=high_ratio;
  out_features[6]=bpoe;
}
