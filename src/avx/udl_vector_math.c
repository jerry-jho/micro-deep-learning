#include <udl.h>
#include <immintrin.h>

#define __VEC_COUNT 8

m_t mac_mmm(m_t * a, m_t * b, l_t size) {
  m_t r = 0;
  float i0[__VEC_COUNT] __attribute__((aligned (32)));
  float i1[__VEC_COUNT] __attribute__((aligned (32)));
  float  o[__VEC_COUNT] __attribute__((aligned (32)));
  l_t idx = 0;
  for (l_t i=0;i<size;i+=__VEC_COUNT) {
    for (l_t j=0;j<__VEC_COUNT;j++) {
      if (idx < size) {
        i0[j] = a[idx];
        i1[j] = b[idx];
      } else {
        i0[j] = 0;
        i1[j] = 0;
      }
      idx++;
    }
    __m256 v0 = _mm256_load_ps(i0);
    __m256 v1 = _mm256_load_ps(i1);
    __m256 v  = _mm256_mul_ps(v0, v1);
    _mm256_store_ps(o, v);
    for (l_t j=0;j<__VEC_COUNT;j++) {
      r = add_mmm(r, o[j]);
    }
  }
  return r;
}
