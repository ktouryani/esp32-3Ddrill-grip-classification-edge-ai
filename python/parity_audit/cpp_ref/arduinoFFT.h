// src/arduinoFFT.h  (desktop-only minimal shim)
#pragma once
#include <cmath>
#include <cstdint>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// API enums used by your code
#define FFT_FORWARD 0
#define FFT_WIN_TYP_HANN 0

template<typename T>
class ArduinoFFT {
public:
  ArduinoFFT() {}

  void windowing(T* vReal, uint16_t samples, uint8_t /*windowType*/, uint8_t /*dir*/) {
    // Hann window
    const int N = samples;
    for (int n = 0; n < N; ++n) {
      T w = (T)(0.5 * (1.0 - std::cos(2.0 * M_PI * n / (N - 1))));
      vReal[n] *= w;
    }
  }

  void compute(T* vReal, T* vImag, uint16_t samples, uint8_t /*dir*/) {
    // Naive DFT (good enough for N=128)
    const int N = samples;
    std::vector<T> R(N, 0), I(N, 0);
    for (int k = 0; k < N; ++k) {
      double re = 0.0, im = 0.0;
      for (int n = 0; n < N; ++n) {
        double a = -2.0 * M_PI * k * n / N;
        re += vReal[n] * std::cos(a) - vImag[n] * std::sin(a);
        im += vReal[n] * std::sin(a) + vImag[n] * std::cos(a);
      }
      R[k] = (T)re;
      I[k] = (T)im;
    }
    for (int k = 0; k < N; ++k) { vReal[k] = R[k]; vImag[k] = I[k]; }
  }

  void complexToMagnitude(T* vReal, T* vImag, uint16_t samples) {
    for (int i = 0; i < samples; ++i) {
      vReal[i] = (T)std::sqrt((double)vReal[i]*vReal[i] + (double)vImag[i]*vImag[i]);
      vImag[i] = 0;
    }
  }
};
