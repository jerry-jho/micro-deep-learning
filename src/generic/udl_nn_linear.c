#include <udl.h>

result_t udl_add_layer(active_t active,tensor4d_t * ifm0, tensor4d_t * ifm1, tensor4d_t * ofm) {
  fmm_t f = get_active_function(active);
  if (f == NULL) return RESULT_ERROR_ACTIVE_NOT_SUPPORT;
  s_t size_0 = tensor4d_size(ifm0);
  s_t size_1 = tensor4d_size(ifm1);
  s_t size_o = tensor4d_size(ofm);
  if (size_0 != size_1) return RESULT_ERROR_SHAPE_MISMATCH;
  if (size_0 != size_o) return RESULT_ERROR_SHAPE_MISMATCH;
  vec_mmm(add_mmm, ifm0->m, ifm1->m, ofm->m, size_0);
  vec_mm(f, ofm->m, ofm->m, size_0);
  return RESULT_OK;
}
