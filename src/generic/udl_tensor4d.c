#include <udl.h>

static void __udl_tensor_init(tensor4d_t * t, content_t content) {
  s_t size = tensor4d_size(t);
  if (t->ttype == type_main) {
    if (content == content_zeros) {
      for (s_t i=0;i<size;i++) {
        t->m[i] = 0;
      }
    } else if (content == content_ones) {
      for (s_t i=0;i<size;i++) {
        t->m[i] = one_m();
      }      
    } else if (content == content_linspace) {
      for (s_t i=0;i<size;i++) {
        t->m[i] = int_m(i);
      }      
    } 
  } else {
    if (content == content_zeros) {
      for (s_t i=0;i<size;i++) {
        t->a[i] = 0;
      }
    } else if (content == content_ones) {
      for (s_t i=0;i<size;i++) {
        t->a[i] = one_a();
      }      
    } else if (content == content_vbs) {
      for (s_t i=0;i<t->shape[3];i++) {
        t->a[i] = 0;
        t->a[i+t->shape[3]] = one_a();
      }      
    }     
  }  
}

tensor4d_t udl_tensor_create_alike(tensor4d_t * s, content_t content) {
  tensor4d_t t;
  t.shape[0] = s->shape[0];
  t.shape[1] = s->shape[1];
  t.shape[2] = s->shape[2];
  t.shape[3] = s->shape[3];
  t.ttype = s->ttype;
  s_t size = tensor4d_size(&t);
  t.m = NULL;
  t.a = NULL;
  if (t.ttype == type_main) {
    t.m = (m_t *)udl_malloc(size * sizeof(m_t));
  } else {
    t.a = (a_t *)udl_malloc(size * sizeof(a_t));
  }
  __udl_tensor_init(&t, content);
  return t;
}

tensor4d_t udl_tensor_create(s_t B, s_t H, s_t W, s_t C, type_t ttype, content_t content) {
  tensor4d_t t;
  t.shape[0] = B;
  t.shape[1] = H;
  t.shape[2] = W;
  t.shape[3] = C;
  t.ttype = ttype;
  return udl_tensor_create_alike(&t, content);
}

tensor4d_t udl_tensor_load(s_t B, s_t H, s_t W, s_t C, type_t ttype, void * source) {
  tensor4d_t t;
  t.shape[0] = B;
  t.shape[1] = H;
  t.shape[2] = W;
  t.shape[3] = C;
  t.ttype = ttype;
  t.m = NULL;
  t.a = NULL;
  if (ttype == type_main) {
    udl_load((void **)&t.m, sizeof(m_t), tensor4d_size(&t), source);
  } else {
    udl_load((void **)&t.a, sizeof(a_t), tensor4d_size(&t), source);
  }
  return t;
}

tensor4d_t udl_tensor_copy(tensor4d_t * s) {
  tensor4d_t t;
  t.shape[0] = s->shape[0];
  t.shape[1] = s->shape[1];
  t.shape[2] = s->shape[2];
  t.shape[3] = s->shape[3];
  t.ttype = s->ttype;
  s_t size = tensor4d_size(&t);
  if (t.ttype == type_main) {
    t.m = (m_t *)udl_malloc(size * sizeof(m_t));
    udl_memcpy(t.m, s->m, size * sizeof(m_t));
  } else {
    t.a = (a_t *)udl_malloc(size * sizeof(a_t));
    udl_memcpy(t.a, s->a, size * sizeof(a_t));
  }
  return t;
}

result_t udl_tensor_reshape(tensor4d_t * s, s_t B, s_t H, s_t W, s_t C) {
  if (tensor4d_size(s) != (B*H*W*C)) return RESULT_ERROR_SHAPE_MISMATCH;
  s->shape[0] = B;
  s->shape[1] = H;
  s->shape[2] = W;
  s->shape[3] = C;  
  return RESULT_OK;
}

s_t udl_tensor_argmax(tensor4d_t * s) {
  if (s->ttype == type_main) {
    if (tensor4d_size(s) > 0) {
      s_t mid = 0;
      m_t m = s->m[0];
      for (s_t id=0;id<tensor4d_size(s);id++) {
        if (s->m[id] > m) {
          mid = id;
          m = s->m[id];
        }
      }
      return mid;
    }
  }
  return 0;
}

void udl_tensor_print(tensor4d_t * t) {
  udl_printf("=== tensor[%d,%d,%d,%d] ===\n",
   (int)t->shape[0], (int)t->shape[1], (int)t->shape[2], (int)t->shape[3]);
  s_t size = tensor4d_size(t);
  udl_m_print(t->m, size);
  udl_printf("\n");
}

void udl_m_print(m_t * m, uint64_t size) {
  for (s_t i=0;i<size;i++) {
    udl_printf("%f ",(double)m[i]);
  }
  udl_printf("\n");
}