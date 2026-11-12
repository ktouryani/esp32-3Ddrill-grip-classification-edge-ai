#include "features.h"
#include <cmath>
#include <numeric>
#include <vector>
#include <algorithm>
#include "arduinoFFT.h" // We still need this for frequency features


// ----- helpers -----
static inline float mean(const float* x, int n) {
  float s = 0.f; for (int i=0;i<n;i++) s += x[i]; return s / n;
}
static inline void remove_dc(float* x, int n) {
  float m = mean(x,n); for (int i=0;i<n;i++) x[i] -= m;
}

// Compute rFFT magnitude into mag[0..K], where K = N/2 (inclusive like numpy rfft)
static void rfft_magnitude_rect(const float* x_in, float* mag_out, int N) {
  // Rectangular window (no windowing) to match Python
  // ArduinoFFT uses in-place buffers; we’ll copy then use its API without windowing.
  static ArduinoFFT<float> FFT;
  static float vReal[WIN_SAMPLES];
  static float vImag[WIN_SAMPLES];

  for (int i=0;i<N;i++){ vReal[i]=x_in[i]; vImag[i]=0.f; }

  // NO Hann: match Python rectangular window
  // FFT.windowing(vReal, N, FFT_WIN_TYP_RECTANGLE, FFT_FORWARD);
  FFT.compute(vReal, vImag, N, FFT_FORWARD);
  FFT.complexToMagnitude(vReal, vImag, N);

  // ArduinoFFT returns N bins (0..N-1) mirrored; rfft wants 0..N/2 inclusive.
  // We will take 0..N/2 exactly.
  int K = N/2;
  for (int k=0;k<=K;k++) mag_out[k] = vReal[k];
}

static float spectral_entropy_norm(const float* x, int N) {
  // P = |rfft(x-mean)|^2; normalize; H / log(len(P))
  static float a[WIN_SAMPLES];
  for (int i=0;i<N;i++) a[i]=x[i];
  remove_dc(a,N);

  // rfft magnitude 0..N/2
  static float M[WIN_SAMPLES/2+1];
  rfft_magnitude_rect(a, M, N);

  // power spectrum
  int K = N/2;
  double tot = 0.0;
  static double P[WIN_SAMPLES/2+1];
  for (int k=0;k<=K;k++){
    double p = (double)M[k]*(double)M[k];
    P[k] = p;
    tot += p;
  }
  if (tot <= 1e-12) tot = 1e-12;

  double H = 0.0;
  for (int k=0;k<=K;k++){
    double q = P[k]/tot;
    if (q>0) H -= q*std::log(q+1e-12);
  }
  return (float)(H / std::log((double)(K+1)));
}

static void spectral_stats_basic_like_py(const float* a, int N, float fs,
                                         float f0, float& centroid_norm, float& high_ratio) {
  // Python uses S = |rfft(a-mean)| (magnitude, NOT power)
  static float x[WIN_SAMPLES];
  for (int i=0;i<N;i++) x[i]=a[i];
  remove_dc(x,N);

  static float S[WIN_SAMPLES/2+1];
  rfft_magnitude_rect(x, S, N);

  int K = N/2;
  const float df = fs / N;

  // centroid using magnitude
  double num=0.0, den=0.0;
  for (int k=1;k<=K;k++){ // skip DC like Python’s max(S[1:])
    double fk = k*df;
    num += fk * (double)S[k];
    den += (double)S[k];
  }
  float centroid = (den>1e-9) ? (float)(num/den) : 0.f;
  centroid_norm = centroid / (fs/2.f);

  // high_ratio: sum S in high (>=2.5*f0) divided by sum S in band (f0..3*f0)
  int k_lo_band = (int)std::ceil(f0/df);
  int k_hi_band = (int)std::floor(3.f*f0/df);
  int k_lo_high = (int)std::ceil(2.5f*f0/df);

  double sum_band=0.0, sum_high=0.0;
  for (int k=1;k<=K;k++){
    double val = (double)S[k];
    if (k>=k_lo_high) sum_high += val;
    if (k>=k_lo_band && k<=k_hi_band) sum_band += val;
  }
  high_ratio = (sum_band>1e-9) ? (float)(sum_high/sum_band) : 0.f;
}

static float band_energy_ratio_power(const float* a, int N, float fs, float f_lo, float f_hi){
  // Python: uses POWER spectrum Xpow_all = |rfft(a-mean)|^2 and returns band/total
  static float x[WIN_SAMPLES];
  for (int i=0;i<N;i++) x[i]=a[i];
  remove_dc(x,N);

  static float M[WIN_SAMPLES/2+1];
  rfft_magnitude_rect(x, M, N);

  int K = N/2;
  const float df = fs / N;
  int k_lo = (int)std::ceil(f_lo/df);
  int k_hi = (int)std::floor(f_hi/df);

  double total=0.0, band=0.0;
  for (int k=0;k<=K;k++){
    double p = (double)M[k]*(double)M[k]; // power
    total += p;
    if (k>=k_lo && k<=k_hi) band += p;
  }
  if (total<=1e-12) return 0.f;
  return (float)(band/total);
}

static float bandpower_over_entropy_like_py(const float* a, int N, float fs, float f0, float delta){
  // Python:
  // Xpow_all = |rfft(a-mean)|^2
  // band_pow = band_energy_ratio(a, f0-2*delta, f0+2*delta) * sum(Xpow_all)
  // Hn = spectral_entropy_norm(a)
  // return band_pow / Hn
  static float x[WIN_SAMPLES];
  for (int i=0;i<N;i++) x[i]=a[i];
  remove_dc(x,N);

  static float M[WIN_SAMPLES/2+1];
  rfft_magnitude_rect(x, M, N);

  int K = N/2;
  const float df = fs / N;

  double total=0.0;
  for (int k=0;k<=K;k++){
    double p = (double)M[k]*(double)M[k];
    total += p;
  }
  if (total<=1e-12) total=1e-12;

  float ber = band_energy_ratio_power(a, N, fs, f0-2.f*delta, f0+2.f*delta); // band/total (power)
  double band_pow = (double)ber * total;

  float Hn = spectral_entropy_norm(a, N);

  return (Hn>1e-12f) ? (float)(band_pow / Hn) : 0.f;
}

void compute_features(const float imu[WIN_SAMPLES][IMU_AXES], float out_features[FEAT_DIM]) {
  // amplitude
  static float a[WIN_SAMPLES];
  for (int i=0;i<WIN_SAMPLES;i++){
    float x=imu[i][0], y=imu[i][1], z=imu[i][2];
    a[i] = std::sqrt(x*x + y*y + z*z);
  }

  // jerk_rms (match Python: prepend a[0] so first diff = 0, mean over N)
  double sumsq = 0.0;
  for (int i=0;i<WIN_SAMPLES;i++){
    float prev = (i==0) ? a[0] : a[i-1];
    float j = (a[i] - prev) * FS_HZ;
    sumsq += (double)j*j;
  }
  float jerk_rms = std::sqrt((float)(sumsq / WIN_SAMPLES));

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

  // spectral stats like Python
  float cen_n=0.f, high_ratio=0.f;
  spectral_stats_basic_like_py(a, WIN_SAMPLES, FS_HZ, F0_HZ, cen_n, high_ratio);

  // ber_f0 using power
  float ber_f0 = band_energy_ratio_power(a, WIN_SAMPLES, FS_HZ, F0_HZ-10.f, F0_HZ+10.f);

  // bandpower_over_entropy_amp like Python (absolute band power / entropy)
  float bpoe = bandpower_over_entropy_like_py(a, WIN_SAMPLES, FS_HZ, F0_HZ, DELTA_HZ);

  // crest factor on amplitude
  double acc=0.0; float mx=0.f;
  for (int i=0;i<WIN_SAMPLES;i++){ acc += (double)a[i]*a[i]; mx = std::max(mx, std::fabs(a[i])); }
  float rms = std::sqrt((float)(acc/WIN_SAMPLES));
  float cf = mx / (rms + 1e-12f);

  // order: ["jerk_rms","centroid_norm","zcr","crest_factor_amp","ber_f0","high_ratio","bandpower_over_entropy_amp"]
  out_features[0]=jerk_rms;
  out_features[1]=cen_n;
  out_features[2]=zcr;
  out_features[3]=cf;
  out_features[4]=ber_f0;
  out_features[5]=high_ratio;
  out_features[6]=bpoe;
}
