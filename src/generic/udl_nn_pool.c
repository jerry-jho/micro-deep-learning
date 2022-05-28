#include <udl.h>

result_t udl_maxpooling2d_layer(maxpooling2d_desc_t desc, tensor4d_t * ofm, tensor4d_t * ifm, tensor4d_t *vbs) {
  s_t OH = ofm->shape[H_];
  s_t OW = ofm->shape[W_];
  s_t OC = ofm->shape[C_];
  s_t IH = ifm->shape[H_];
  s_t IW = ifm->shape[W_];
  s_t IC = ifm->shape[C_];
  s_t B  = ifm->shape[B_];
  s_t KH = desc.kh;
  s_t KW = desc.kw;

  if (B  != ofm->shape[B_]) return RESULT_ERROR_SHAPE_MISMATCH;
  if (IC != OC) return RESULT_ERROR_SHAPE_MISMATCH;
  
  for (s_t b=0;b<B;b++) {
    for (s_t oz=0;oz<OC;oz++) {
      for (s_t oy=0;oy<OH;oy++) {
        for (s_t ox=0;ox<OW;ox++) {
          //printf("oy = %d ox = %d oz = %d\n", oy, ox, oz);
          s_t o_index = b*OH*OW*OC+oy*OW*OC+ox*OC+oz;
          int first = 1;
          m_t v;
          for (s_t ky=0;ky<KH;ky++) {
            for (s_t kx=0;kx<KW;kx++) {
              s_t iy = oy * desc.sy + ky;
              s_t ix = ox * desc.sx + kx;
              if (iy < desc.py || iy >= (IH + desc.py) ||
                  ix < desc.px || ix >= (IW + desc.px)) {
                v = 0;
              } else {
                iy -= desc.py;
                ix -= desc.px;
                s_t i_index = b*IH*IW*IC+iy*IW*IC+ix*IC+oz;
                v = ifm->m[i_index];
              }
              if (first) {
                ofm->m[o_index] = v;
                first = 0;
              } else {
                ofm->m[o_index] = max_mmm(ofm->m[o_index], v);
              }
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

result_t udl_avgpooling2d_layer(avgpooling2d_desc_t desc, tensor4d_t * ofm, tensor4d_t * ifm, tensor4d_t *vbs) {
  s_t OH = ofm->shape[H_];
  s_t OW = ofm->shape[W_];
  s_t OC = ofm->shape[C_];
  s_t IH = ifm->shape[H_];
  s_t IW = ifm->shape[W_];
  s_t IC = ifm->shape[C_];
  s_t B  = ifm->shape[B_];
  s_t KH = desc.kh;
  s_t KW = desc.kw;

  if (B  != ofm->shape[B_]) return RESULT_ERROR_SHAPE_MISMATCH;
  if (IC != OC) return RESULT_ERROR_SHAPE_MISMATCH;
  
  for (s_t b=0;b<B;b++) {
    for (s_t oz=0;oz<OC;oz++) {
      for (s_t oy=0;oy<OH;oy++) {
        for (s_t ox=0;ox<OW;ox++) {
          //printf("oy = %d ox = %d oz = %d\n", oy, ox, oz);
          s_t o_index = b*OH*OW*OC+oy*OW*OC+ox*OC+oz;
          m_t v = 0;
          s_t cnt = 0;
          for (s_t ky=0;ky<KH;ky++) {
            for (s_t kx=0;kx<KW;kx++) {
              s_t iy = oy * desc.sy + ky;
              s_t ix = ox * desc.sx + kx;
              if (iy < desc.py || iy >= (IH + desc.py) ||
                  ix < desc.px || ix >= (IW + desc.px)) {
                v = 0;
              } else {
                iy -= desc.py;
                ix -= desc.px;
                s_t i_index = b*IH*IW*IC+iy*IW*IC+ix*IC+oz;
                v = add_mmm(v, ifm->m[i_index]);
                cnt ++;
              }
            }
          }
          ofm->m[o_index] = div_mim(v, cnt); 
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