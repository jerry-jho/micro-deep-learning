#include <udl.h>

result_t max_error(tensor4d_t * ifm0, tensor4d_t * ifm1, m_t * r) {
  s_t size_0 = tensor4d_size(ifm0);
  s_t size_1 = tensor4d_size(ifm1);
  if (size_0 != size_1) return RESULT_ERROR_SHAPE_MISMATCH;
  if (size_0 == 0) return RESULT_ERROR_SHAPE_ZERO;
  m_t m = 0;
  for (s_t i=0;i<size_0;i++) {
    m_t e = abs_mmm(sub_mmm(ifm0->m[i], ifm1->m[i]));
    m = max_mmm(m, e);
  }
  * r = m;
  return RESULT_OK;
}
