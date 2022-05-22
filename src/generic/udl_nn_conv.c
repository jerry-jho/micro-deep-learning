#include <udl.h>

result_t udl_conv2d_layer(conv2d_desc_t desc,tensor4d_t * ofm, tensor4d_t * ifm, tensor4d_t * ker, tensor4d_t *vbs) {
  s_t OH = ofm->shape[H_];
  s_t OW = ofm->shape[W_];
  s_t OC = ofm->shape[C_];
  s_t IH = ifm->shape[H_];
  s_t IW = ifm->shape[W_];
  s_t IC = ifm->shape[C_];
  s_t KH = ker->shape[H_];
  s_t KW = ker->shape[W_];
  s_t B  = ifm->shape[B_];
/*
  printf("OC = %d ker->shape[N_] = %d\n", OC, ker->shape[N_]);
  printf("IC = %d ker->shape[C_] = %d\n", IC, ker->shape[C_]);
  printf("B  = %d ker->shape[B_] = %d\n", B , ofm->shape[B_]);
*/
  if (OC != ker->shape[N_]) return RESULT_ERROR_SHAPE_MISMATCH;
  if (IC != ker->shape[C_]) return RESULT_ERROR_SHAPE_MISMATCH;
  if (B  != ofm->shape[B_]) return RESULT_ERROR_SHAPE_MISMATCH;
  
  for (s_t b=0;b<B;b++) {
    for (s_t oz=0;oz<OC;oz++) {
      for (s_t oy=0;oy<OH;oy++) {
        for (s_t ox=0;ox<OW;ox++) {
          //printf("oy = %d ox = %d oz = %d\n", oy, ox, oz);
          s_t o_index = b*OH*OW*OC+oy*OW*OC+ox*OC+oz;
          ofm->m[o_index] = 0;
          for (s_t ky=0;ky<KH;ky++) {
            for (s_t kx=0;kx<KW;kx++) {
              s_t iy = oy * desc.sy + ky;
              s_t ix = ox * desc.sx + kx;
              if (iy < desc.py || iy >= (IH + desc.py) ||
                  ix < desc.px || ix >= (IW + desc.px)) {
                continue;
              }
              
              iy -= desc.py;
              ix -= desc.px;
              //printf("ky = %d kx = %d iy = %d ix = %d ", ky, kx, iy, ix);
              m_t m = mac_mmm(
                ifm->m + (b*IH*IW*IC+iy*IW*IC+ix*IC),
                ker->m + (oz*KH*KW*IC+ky*KW*IC+kx*IC), IC);
              //printf("m = %f\n", m);
              ofm->m[o_index] += m;
            }
          }
          if (vbs != NULL) {
            ofm->m[o_index] = mac_mam(
              ofm->m[o_index],
              vbs->a[oz],
              vbs->a[oz+OC]);
          }
          if (desc.active == active_type_relu) {
            ofm->m[o_index] = active_function_relu(ofm->m[o_index]);
          }
          //printf("%d %d %d %f\n", oy, ox, oz, ofm->m[o_index]);
        }
      }
    }
  }
  return RESULT_OK;
}

result_t udl_dense_layer(dense_desc_t desc,tensor4d_t * ofm, tensor4d_t * ifm, tensor4d_t * ker, tensor4d_t *vbs) {
  s_t OH = ofm->shape[H_];
  s_t OW = ofm->shape[W_];
  s_t OC = ofm->shape[C_];
  s_t IH = ifm->shape[H_];
  s_t IW = ifm->shape[W_];
  s_t IC = ifm->shape[C_];
  s_t KH = ker->shape[H_];
  s_t KW = ker->shape[W_];
  s_t B  = ifm->shape[B_];
/*
  printf("OC = %d ker->shape[N_] = %d\n", OC, ker->shape[N_]);
  printf("IC = %d ker->shape[C_] = %d\n", IC, ker->shape[C_]);
  printf("B  = %d ker->shape[B_] = %d\n", B , ofm->shape[B_]);
*/
  if (OH != 1) return RESULT_ERROR_SHAPE_MISMATCH;
  if (OW != 1) return RESULT_ERROR_SHAPE_MISMATCH;
  if (IH != 1) return RESULT_ERROR_SHAPE_MISMATCH;
  if (IW != 1) return RESULT_ERROR_SHAPE_MISMATCH;
  if (KH != 1) return RESULT_ERROR_SHAPE_MISMATCH;
  if (KW != 1) return RESULT_ERROR_SHAPE_MISMATCH;
  if (OC != ker->shape[N_]) return RESULT_ERROR_SHAPE_MISMATCH;
  if (IC != ker->shape[C_]) return RESULT_ERROR_SHAPE_MISMATCH;
  if (B  != ofm->shape[B_]) return RESULT_ERROR_SHAPE_MISMATCH;
  
  for (s_t b=0;b<B;b++) {
    for (s_t oz=0;oz<OC;oz++) {
      s_t o_index = b*OC+oz;
      ofm->m[o_index] = 0;
      m_t m = mac_mmm(
         ifm->m + (b * IC),
         ker->m + (oz * IC), IC);
      ofm->m[o_index] = m;
      if (vbs != NULL) {
        ofm->m[o_index] = mac_mam(
          ofm->m[o_index],
          vbs->a[oz],
          vbs->a[oz+OC]);
      }
      if (desc.active == active_type_relu) {
        ofm->m[o_index] = active_function_relu(ofm->m[o_index]);
      }
    }
  }
  return RESULT_OK;
}
