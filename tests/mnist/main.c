#include <udl.h>

#include "cdata/data.c"

int main(int argc, char ** argv) {
  tensor4d_t im_data      = udl_tensor_from_buffer(1,28,28,1,type_main,im_data_bin);
  tensor4d_t conv0_weight = udl_tensor_from_buffer(16,5,5,1,type_main, conv0_0_weight);
  tensor4d_t conv0_bias   = udl_tensor_from_buffer(1,1,2,16,type_aux, conv0_0_bias);
  tensor4d_t conv0_out    = udl_tensor_create(1,28,28,16,type_main,content_zeros);
  tensor4d_t pool0_out    = udl_tensor_create(1,14,14,16,type_main,content_zeros);
  tensor4d_t conv1_weight = udl_tensor_from_buffer(32,5,5,16,type_main,conv1_0_weight);
  tensor4d_t conv1_bias   = udl_tensor_from_buffer(1,1,2,32,type_aux,conv1_0_bias);
  tensor4d_t conv1_out    = udl_tensor_create(1,14,14,32,type_main,content_zeros);
  tensor4d_t pool1_out    = udl_tensor_create(1,7,7,32,type_main,content_zeros);
  tensor4d_t dense_weight = udl_tensor_from_buffer(10,1,1,7*7*32,type_main,out_weight);
  tensor4d_t dense_bias   = udl_tensor_from_buffer(1,1,2,10,type_aux,out_bias);
  tensor4d_t dense_out    = udl_tensor_create(1,1,1,10,type_main,content_zeros);
  conv2d_desc_t conv0_desc = {.active = active_type_relu, .px = 2, .py = 2, .sx = 1, .sy = 1};
  maxpooling2d_desc_t pool0_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 2, .sy = 2, .kh = 2, .kw = 2};
  conv2d_desc_t conv1_desc = {.active = active_type_relu, .px = 2, .py = 2, .sx = 1, .sy = 1};
  maxpooling2d_desc_t pool1_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 2, .sy = 2, .kh = 2, .kw = 2};
  dense_desc_t dense_desc = {.active = active_type_linear};
  udl_conv2d_layer(conv0_desc, &conv0_out, &im_data, &conv0_weight, &conv0_bias);
  udl_maxpooling2d_layer(pool0_desc, &pool0_out, &conv0_out,  NULL);
  udl_conv2d_layer(conv1_desc, &conv1_out, &pool0_out, &conv1_weight, &conv1_bias);
  udl_maxpooling2d_layer(pool1_desc, &pool1_out, &conv1_out,  NULL);
  udl_tensor_reshape(&pool1_out, 1, 1, 1, 7*7*32);
  udl_dense_layer(dense_desc, &dense_out, &pool1_out, &dense_weight, &dense_bias);
  udl_tensor_print(&dense_out);
  udl_printf("%d\n", udl_tensor_argmax(&dense_out));
  return 0;
}