#ifndef __UDL_H
#define __UDL_H

/* C Basic */
#ifndef NULL
#define NULL ((void *)0)
#endif

typedef unsigned int s_t;
typedef unsigned long long l_t;

void * udl_malloc(l_t size);
void udl_free(void * v);
void udl_memcpy(void * dst, void * src, l_t size);
/* API Basic */

typedef int result_t;
#define RESULT_OK                         0
#define RESULT_ERROR_SHAPE_MISMATCH     -21
#define RESULT_ERROR_SHAPE_ZERO         -22
#define RESULT_ERROR_ACTIVE_NOT_SUPPORT -10
#define RESULT_ERROR_FILE_NOT_FOUND    -255
#define RESULT_ERROR_FUNCTION_NOT_EXIST  -1

/* IO Basic */
void udl_putc(char c);
int udl_puts(char * s);
result_t udl_load(void ** dest, l_t trunk, l_t count, void * source);
int udl_printf(const char *str, ...);

/* Math Basic */
#define UDL_SYMBOL_TYPE_FP32 0
#define UDL_SYMBOL_TYPE UDL_SYMBOL_TYPE_FP32

#if UDL_SYMBOL_TYPE == UDL_SYMBOL_TYPE_FP32
typedef float m_t;
typedef float a_t;
#endif
m_t one_m();
a_t one_a();
m_t int_m(int x);
m_t mul_mmm(m_t a, m_t b);
a_t mul_maa(m_t a, a_t b);
m_t mul_mam(m_t a, a_t b);
m_t div_mim(m_t a, s_t b);
m_t div_mmm(m_t a, m_t b);
m_t mac_mam(m_t a, a_t s, a_t b);
m_t add_mmm(m_t a, m_t b);
m_t sub_mmm(m_t a, m_t b);
m_t abs_mmm(m_t a);
m_t max_mmm(m_t a, m_t b);
m_t min_mmm(m_t a, m_t b);
m_t exp_mmm(m_t a);

/* Vector Math Basic */
typedef m_t (* fmm_t)(m_t);
typedef m_t (* fmmm_t)(m_t, m_t);
void vec_mm(fmm_t f, m_t * a, m_t * c, l_t size);
void vec_mmm(fmmm_t f, m_t * a, m_t * b, m_t * c, l_t size);
m_t mac_mmm(m_t * a, m_t * b, l_t size);
/* Tensor Basic */



typedef enum {
  type_main,
  type_aux
} type_t;

#define B_ 0
#define N_ 0
#define H_ 1
#define W_ 2
#define C_ 3

typedef struct _tensor4d_t {
  // for featuremap, B-H-W-C
  // for kernel      N-KH-KW-C
  s_t   shape[4];
  m_t * m;
  a_t * a;
  type_t ttype;
} tensor4d_t;

#define tensor4d_size(t) ((t)->shape[0]*(t)->shape[1]*(t)->shape[2]*(t)->shape[3])

typedef enum {
  content_empty,
  content_zeros,
  content_ones,
  content_linspace,
  content_vbs
} content_t;



tensor4d_t udl_tensor_create(s_t B, s_t H, s_t W, s_t C, type_t ttype, content_t content);
tensor4d_t udl_tensor_create_alike(tensor4d_t * s, content_t content);
tensor4d_t udl_tensor_copy(tensor4d_t * s);
tensor4d_t udl_tensor_load(s_t B, s_t H, s_t W, s_t C, type_t ttype, void * source);
tensor4d_t udl_tensor_from_buffer(s_t B, s_t H, s_t W, s_t C, type_t ttype, void * source);
result_t udl_tensor_reshape(tensor4d_t * s, s_t B, s_t H, s_t W, s_t C);
s_t udl_tensor_argmax(tensor4d_t * s);
void udl_tensor_print(tensor4d_t * t);
void udl_m_print(m_t * t, l_t size);
/* Active Basic */

typedef enum {
  active_type_linear,
  active_type_relu
} active_t;


fmm_t get_active_function(active_t active);
m_t active_function_relu(m_t a);
m_t active_function_linear(m_t a);


/* NN */

typedef struct _desc_t {
  s_t kh;
  s_t kw;
  s_t py;
  s_t px;
  s_t sy;
  s_t sx;
  active_t active;
} desc_t;

typedef desc_t conv2d_desc_t;
typedef desc_t maxpooling2d_desc_t;
typedef desc_t avgpooling2d_desc_t;
typedef desc_t batch_normalization_desc_t;
typedef desc_t dense_desc_t;
typedef desc_t add_desc_t;

result_t udl_conv2d_layer(conv2d_desc_t desc,tensor4d_t * ofm, tensor4d_t * ifm, tensor4d_t * ker, tensor4d_t *vbs);
result_t udl_batch_normalization_layer(batch_normalization_desc_t desc, tensor4d_t * ofm, tensor4d_t * ifm, tensor4d_t *vbs);
result_t udl_dense_layer(dense_desc_t desc,tensor4d_t * ofm, tensor4d_t * ifm, tensor4d_t * ker, tensor4d_t *vbs);
result_t udl_maxpooling2d_layer(maxpooling2d_desc_t desc, tensor4d_t * ofm, tensor4d_t * ifm, tensor4d_t *vbs);
result_t udl_avgpooling2d_layer(avgpooling2d_desc_t desc, tensor4d_t * ofm, tensor4d_t * ifm, tensor4d_t *vbs);
result_t udl_add_layer(add_desc_t desc, tensor4d_t * ofm, tensor4d_t * ifm0, tensor4d_t * ifm1);
result_t udl_softmax_layer(tensor4d_t * ofm, tensor4d_t * ifm);

#endif