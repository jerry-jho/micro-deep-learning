#include <udl.h>

void vec_mm(fmm_t f, m_t * a, m_t * c, l_t size) {
  for (l_t i=0;i<size;i++) {
    c[i] = f(a[i]);
  }
}

void vec_mmm(fmmm_t f, m_t * a, m_t * b, m_t * c, l_t size) {
  for (l_t i=0;i<size;i++) {
    c[i] = f(a[i], b[i]);
  }
}

