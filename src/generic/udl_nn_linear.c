#include <udl.h>

result_t udl_add_layer(add_desc_t desc,tensor4d_t * ofm, tensor4d_t * ifm0, tensor4d_t * ifm1) {
  fmm_t f = get_active_function(desc.active);
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

result_t udl_batch_normalization_layer(batch_normalization_desc_t desc, tensor4d_t * ofm, tensor4d_t * ifm, tensor4d_t *vbs) {
  s_t OH = ofm->shape[H_];
  s_t OW = ofm->shape[W_];
  s_t OC = ofm->shape[C_];
  s_t IH = ifm->shape[H_];
  s_t IW = ifm->shape[W_];
  s_t IC = ifm->shape[C_];
  s_t B  = ifm->shape[B_];
/*
  printf("OC = %d ker->shape[N_] = %d\n", OC, ker->shape[N_]);
  printf("IC = %d ker->shape[C_] = %d\n", IC, ker->shape[C_]);
  printf("B  = %d ker->shape[B_] = %d\n", B , ofm->shape[B_]);
*/
  if (OC != IC) return RESULT_ERROR_SHAPE_MISMATCH;
  if (OH != IH) return RESULT_ERROR_SHAPE_MISMATCH;
  if (OW != IW) return RESULT_ERROR_SHAPE_MISMATCH;  
  if (B  != ofm->shape[B_]) return RESULT_ERROR_SHAPE_MISMATCH;
  
  for (s_t b=0;b<B;b++) {
    for (s_t oz=0;oz<OC;oz++) {
      for (s_t oy=0;oy<OH;oy++) {
        for (s_t ox=0;ox<OW;ox++) {
          s_t o_index = b*OH*OW*OC+oy*OW*OC+ox*OC+oz;
          ofm->m[o_index] = 0;
          if (vbs != NULL) {
            ofm->m[o_index] = mac_mam(
              ifm->m[o_index],
              vbs->a[oz],
              vbs->a[oz+OC]);
          }
          if (desc.active == active_type_relu) {
            ofm->m[o_index] = active_function_relu(ofm->m[o_index]);
          }
        }
      }
    }
  }
  return RESULT_OK;
}

result_t udl_softmax_layer(tensor4d_t * ofm, tensor4d_t * ifm) {
  s_t size_o = tensor4d_size(ofm);
  s_t size_i = tensor4d_size(ifm);
  if (size_i != size_o) return RESULT_ERROR_SHAPE_MISMATCH;
  m_t sum = 0;
  for (s_t b=0;b<size_i;b=b+1) {
    ofm->m[b] = exp_mmm(ifm->m[b]);
    sum = add_mmm(sum, ofm->m[b]);
  }
  for (s_t b=0;b<size_i;b=b+1) {
    ofm->m[b] = div_mmm(ofm->m[b], sum);
  }
  return RESULT_OK; 
}