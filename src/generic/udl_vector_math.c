#include <udl.h>

m_t mac_mmm(m_t * a, m_t * b, l_t size) {
  m_t r = 0;
  for (l_t i=0;i<size;i++) {
    r = add_mmm(r, mul_mmm(a[i], b[i]));
  }
  return r;
}
