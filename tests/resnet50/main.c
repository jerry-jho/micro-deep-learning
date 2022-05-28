#include <udl.h>

#include "py/data.h"
#include "py/imagenet.c"

int main(int argc, char ** argv) {
  tensor4d_t t_input              = udl_tensor_from_buffer(1,224,224,3,type_main,input);

  //block1

  tensor4d_t t_conv1_conv_k       = udl_tensor_from_buffer(64,7,7,3,type_main, conv1_conv_k);
  tensor4d_t t_conv1_conv_b       = udl_tensor_from_buffer(1,1,2,64,type_aux, conv1_conv_b);
  conv2d_desc_t conv1_conv_desc = {.active = active_type_linear, .px = 3, .py = 3, .sx = 2, .sy = 2};
  tensor4d_t t_conv1_conv         = udl_tensor_create(1,112,112,64,type_main,content_zeros);
  udl_conv2d_layer(conv1_conv_desc, &t_conv1_conv, &t_input, &t_conv1_conv_k, &t_conv1_conv_b);

  tensor4d_t t_conv1_bn_b       = udl_tensor_from_buffer(1,1,2,64,type_aux, conv1_bn_b);
  batch_normalization_desc_t conv1_bn_desc = {.active = active_type_relu};
  tensor4d_t t_conv1_relu         = udl_tensor_create(1,112,112,64,type_main,content_zeros);
  udl_batch_normalization_layer(conv1_bn_desc, &t_conv1_relu, &t_conv1_conv, &t_conv1_bn_b);  

  maxpooling2d_desc_t pool1_pool_desc = {.active = active_type_linear, .px = 1, .py = 1, .sx = 2, .sy = 2, .kw = 3, .kh = 3};
  tensor4d_t t_pool1_pool         = udl_tensor_create(1,56,56,64,type_main,content_zeros);
  udl_maxpooling2d_layer(pool1_pool_desc, &t_pool1_pool, &t_conv1_relu,  NULL);
  
  //block2-1

  tensor4d_t t_conv2_block1_1_conv_k       = udl_tensor_from_buffer(64,1,1,64,type_main, conv2_block1_1_conv_k);
  tensor4d_t t_conv2_block1_1_conv_b       = udl_tensor_from_buffer(1,1,2,64,type_aux, conv2_block1_1_conv_b);
  conv2d_desc_t conv2_block1_1_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 1, .sy = 1};
  tensor4d_t t_conv2_block1_1_conv         = udl_tensor_create(1,56,56,64,type_main,content_zeros);
  udl_conv2d_layer(conv2_block1_1_conv_desc, &t_conv2_block1_1_conv, &t_pool1_pool, &t_conv2_block1_1_conv_k, &t_conv2_block1_1_conv_b);

  tensor4d_t t_conv2_block1_1_bn_b       = udl_tensor_from_buffer(1,1,2,64,type_aux, conv2_block1_1_bn_b);
  batch_normalization_desc_t conv2_block1_1_bn_desc = {.active = active_type_relu};
  tensor4d_t t_conv2_block1_1_relu         = udl_tensor_create(1,56,56,64,type_main,content_zeros);
  udl_batch_normalization_layer(conv2_block1_1_bn_desc, &t_conv2_block1_1_relu, &t_conv2_block1_1_conv, &t_conv2_block1_1_bn_b); 

  tensor4d_t t_conv2_block1_2_conv_k       = udl_tensor_from_buffer(64,3,3,64,type_main, conv2_block1_2_conv_k);
  tensor4d_t t_conv2_block1_2_conv_b       = udl_tensor_from_buffer(1,1,2,64,type_aux, conv2_block1_2_conv_b);
  conv2d_desc_t conv2_block1_2_conv_desc = {.active = active_type_linear, .px = 1, .py = 1, .sx = 1, .sy = 1};
  tensor4d_t t_conv2_block1_2_conv         = udl_tensor_create(1,56,56,64,type_main,content_zeros);
  udl_conv2d_layer(conv2_block1_2_conv_desc, &t_conv2_block1_2_conv, &t_conv2_block1_1_relu, &t_conv2_block1_2_conv_k, &t_conv2_block1_2_conv_b);

  tensor4d_t t_conv2_block1_2_bn_b       = udl_tensor_from_buffer(1,1,2,64,type_aux, conv2_block1_2_bn_b);
  batch_normalization_desc_t conv2_block1_2_bn_desc = {.active = active_type_relu};
  tensor4d_t t_conv2_block1_2_relu         = udl_tensor_create(1,56,56,64,type_main,content_zeros);
  udl_batch_normalization_layer(conv2_block1_2_bn_desc, &t_conv2_block1_2_relu, &t_conv2_block1_2_conv, &t_conv2_block1_2_bn_b); 

  tensor4d_t t_conv2_block1_3_conv_k       = udl_tensor_from_buffer(256,1,1,64,type_main, conv2_block1_3_conv_k);
  tensor4d_t t_conv2_block1_3_conv_b       = udl_tensor_from_buffer(1,1,2,256,type_aux, conv2_block1_3_conv_b);
  conv2d_desc_t conv2_block1_3_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 1, .sy = 1};
  tensor4d_t t_conv2_block1_3_conv         = udl_tensor_create(1,56,56,256,type_main,content_zeros);
  udl_conv2d_layer(conv2_block1_3_conv_desc, &t_conv2_block1_3_conv, &t_conv2_block1_2_relu, &t_conv2_block1_3_conv_k, &t_conv2_block1_3_conv_b);

  tensor4d_t t_conv2_block1_3_bn_b       = udl_tensor_from_buffer(1,1,2,256,type_aux, conv2_block1_3_bn_b);
  batch_normalization_desc_t conv2_block1_3_bn_desc = {.active = active_type_linear};
  tensor4d_t t_conv2_block1_3_bn         = udl_tensor_create(1,56,56,256,type_main,content_zeros);
  udl_batch_normalization_layer(conv2_block1_3_bn_desc, &t_conv2_block1_3_bn, &t_conv2_block1_3_conv, &t_conv2_block1_3_bn_b); 

  tensor4d_t t_conv2_block1_0_conv_k       = udl_tensor_from_buffer(256,1,1,64,type_main, conv2_block1_0_conv_k);
  tensor4d_t t_conv2_block1_0_conv_b       = udl_tensor_from_buffer(1,1,2,256,type_aux, conv2_block1_0_conv_b);
  conv2d_desc_t conv2_block1_0_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 1, .sy = 1};
  tensor4d_t t_conv2_block1_0_conv         = udl_tensor_create(1,56,56,256,type_main,content_zeros);
  udl_conv2d_layer(conv2_block1_0_conv_desc, &t_conv2_block1_0_conv, &t_pool1_pool, &t_conv2_block1_0_conv_k, &t_conv2_block1_0_conv_b);

  tensor4d_t t_conv2_block1_0_bn_b       = udl_tensor_from_buffer(1,1,2,256,type_aux, conv2_block1_0_bn_b);
  batch_normalization_desc_t conv2_block1_0_bn_desc = {.active = active_type_linear};
  tensor4d_t t_conv2_block1_0_bn         = udl_tensor_create(1,56,56,256,type_main,content_zeros);
  udl_batch_normalization_layer(conv2_block1_0_bn_desc, &t_conv2_block1_0_bn, &t_conv2_block1_0_conv, &t_conv2_block1_0_bn_b); 

  tensor4d_t t_conv2_block1_out         = udl_tensor_create(1,56,56,256,type_main,content_zeros);
  add_desc_t conv2_block1_add_desc      = {.active = active_type_relu};
  udl_add_layer(conv2_block1_add_desc, &t_conv2_block1_out, &t_conv2_block1_3_bn, &t_conv2_block1_0_bn);

  //block2-2

  tensor4d_t t_conv2_block2_1_conv_k       = udl_tensor_from_buffer(64,1,1,256,type_main, conv2_block2_1_conv_k);
  tensor4d_t t_conv2_block2_1_conv_b       = udl_tensor_from_buffer(1,1,2,64,type_aux, conv2_block2_1_conv_b);
  conv2d_desc_t conv2_block2_1_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 1, .sy = 1};
  tensor4d_t t_conv2_block2_1_conv         = udl_tensor_create(1,56,56,64,type_main,content_zeros);
  udl_conv2d_layer(conv2_block2_1_conv_desc, &t_conv2_block2_1_conv, &t_conv2_block1_out, &t_conv2_block2_1_conv_k, &t_conv2_block2_1_conv_b);

  tensor4d_t t_conv2_block2_1_bn_b       = udl_tensor_from_buffer(1,1,2,64,type_aux, conv2_block2_1_bn_b);
  batch_normalization_desc_t conv2_block2_1_bn_desc = {.active = active_type_relu};
  tensor4d_t t_conv2_block2_1_relu         = udl_tensor_create(1,56,56,64,type_main,content_zeros);
  udl_batch_normalization_layer(conv2_block2_1_bn_desc, &t_conv2_block2_1_relu, &t_conv2_block2_1_conv, &t_conv2_block2_1_bn_b); 

  tensor4d_t t_conv2_block2_2_conv_k       = udl_tensor_from_buffer(64,3,3,64,type_main, conv2_block2_2_conv_k);
  tensor4d_t t_conv2_block2_2_conv_b       = udl_tensor_from_buffer(1,1,2,64,type_aux, conv2_block2_2_conv_b);
  conv2d_desc_t conv2_block2_2_conv_desc = {.active = active_type_linear, .px = 1, .py = 1, .sx = 1, .sy = 1};
  tensor4d_t t_conv2_block2_2_conv         = udl_tensor_create(1,56,56,64,type_main,content_zeros);
  udl_conv2d_layer(conv2_block2_2_conv_desc, &t_conv2_block2_2_conv, &t_conv2_block2_1_relu, &t_conv2_block2_2_conv_k, &t_conv2_block2_2_conv_b);

  tensor4d_t t_conv2_block2_2_bn_b       = udl_tensor_from_buffer(1,1,2,64,type_aux, conv2_block2_2_bn_b);
  batch_normalization_desc_t conv2_block2_2_bn_desc = {.active = active_type_relu};
  tensor4d_t t_conv2_block2_2_relu         = udl_tensor_create(1,56,56,64,type_main,content_zeros);
  udl_batch_normalization_layer(conv2_block2_2_bn_desc, &t_conv2_block2_2_relu, &t_conv2_block2_2_conv, &t_conv2_block2_2_bn_b); 

  tensor4d_t t_conv2_block2_3_conv_k       = udl_tensor_from_buffer(256,1,1,64,type_main, conv2_block2_3_conv_k);
  tensor4d_t t_conv2_block2_3_conv_b       = udl_tensor_from_buffer(1,1,2,256,type_aux, conv2_block2_3_conv_b);
  conv2d_desc_t conv2_block2_3_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 1, .sy = 1};
  tensor4d_t t_conv2_block2_3_conv         = udl_tensor_create(1,56,56,256,type_main,content_zeros);
  udl_conv2d_layer(conv2_block2_3_conv_desc, &t_conv2_block2_3_conv, &t_conv2_block2_2_relu, &t_conv2_block2_3_conv_k, &t_conv2_block2_3_conv_b);

  tensor4d_t t_conv2_block2_3_bn_b       = udl_tensor_from_buffer(1,1,2,256,type_aux, conv2_block2_3_bn_b);
  batch_normalization_desc_t conv2_block2_3_bn_desc = {.active = active_type_linear};
  tensor4d_t t_conv2_block2_3_bn         = udl_tensor_create(1,56,56,256,type_main,content_zeros);
  udl_batch_normalization_layer(conv2_block2_3_bn_desc, &t_conv2_block2_3_bn, &t_conv2_block2_3_conv, &t_conv2_block2_3_bn_b); 

  tensor4d_t t_conv2_block2_out         = udl_tensor_create(1,56,56,256,type_main,content_zeros);
  add_desc_t conv2_block2_add_desc      = {.active = active_type_relu};
  udl_add_layer(conv2_block2_add_desc, &t_conv2_block2_out, &t_conv2_block2_3_bn, &t_conv2_block1_out);

  //block2-3

  tensor4d_t t_conv2_block3_1_conv_k       = udl_tensor_from_buffer(64,1,1,256,type_main, conv2_block3_1_conv_k);
  tensor4d_t t_conv2_block3_1_conv_b       = udl_tensor_from_buffer(1,1,2,64,type_aux, conv2_block3_1_conv_b);
  conv2d_desc_t conv2_block3_1_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 1, .sy = 1};
  tensor4d_t t_conv2_block3_1_conv         = udl_tensor_create(1,56,56,64,type_main,content_zeros);
  udl_conv2d_layer(conv2_block3_1_conv_desc, &t_conv2_block3_1_conv, &t_conv2_block2_out, &t_conv2_block3_1_conv_k, &t_conv2_block3_1_conv_b);

  tensor4d_t t_conv2_block3_1_bn_b       = udl_tensor_from_buffer(1,1,2,64,type_aux, conv2_block3_1_bn_b);
  batch_normalization_desc_t conv2_block3_1_bn_desc = {.active = active_type_relu};
  tensor4d_t t_conv2_block3_1_relu         = udl_tensor_create(1,56,56,64,type_main,content_zeros);
  udl_batch_normalization_layer(conv2_block3_1_bn_desc, &t_conv2_block3_1_relu, &t_conv2_block3_1_conv, &t_conv2_block3_1_bn_b); 

  tensor4d_t t_conv2_block3_2_conv_k       = udl_tensor_from_buffer(64,3,3,64,type_main, conv2_block3_2_conv_k);
  tensor4d_t t_conv2_block3_2_conv_b       = udl_tensor_from_buffer(1,1,2,64,type_aux, conv2_block3_2_conv_b);
  conv2d_desc_t conv2_block3_2_conv_desc = {.active = active_type_linear, .px = 1, .py = 1, .sx = 1, .sy = 1};
  tensor4d_t t_conv2_block3_2_conv         = udl_tensor_create(1,56,56,64,type_main,content_zeros);
  udl_conv2d_layer(conv2_block3_2_conv_desc, &t_conv2_block3_2_conv, &t_conv2_block3_1_relu, &t_conv2_block3_2_conv_k, &t_conv2_block3_2_conv_b);

  tensor4d_t t_conv2_block3_2_bn_b       = udl_tensor_from_buffer(1,1,2,64,type_aux, conv2_block3_2_bn_b);
  batch_normalization_desc_t conv2_block3_2_bn_desc = {.active = active_type_relu};
  tensor4d_t t_conv2_block3_2_relu         = udl_tensor_create(1,56,56,64,type_main,content_zeros);
  udl_batch_normalization_layer(conv2_block3_2_bn_desc, &t_conv2_block3_2_relu, &t_conv2_block3_2_conv, &t_conv2_block3_2_bn_b); 

  tensor4d_t t_conv2_block3_3_conv_k       = udl_tensor_from_buffer(256,1,1,64,type_main, conv2_block3_3_conv_k);
  tensor4d_t t_conv2_block3_3_conv_b       = udl_tensor_from_buffer(1,1,2,256,type_aux, conv2_block3_3_conv_b);
  conv2d_desc_t conv2_block3_3_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 1, .sy = 1};
  tensor4d_t t_conv2_block3_3_conv         = udl_tensor_create(1,56,56,256,type_main,content_zeros);
  udl_conv2d_layer(conv2_block3_3_conv_desc, &t_conv2_block3_3_conv, &t_conv2_block3_2_relu, &t_conv2_block3_3_conv_k, &t_conv2_block3_3_conv_b);

  tensor4d_t t_conv2_block3_3_bn_b       = udl_tensor_from_buffer(1,1,2,256,type_aux, conv2_block3_3_bn_b);
  batch_normalization_desc_t conv2_block3_3_bn_desc = {.active = active_type_linear};
  tensor4d_t t_conv2_block3_3_bn         = udl_tensor_create(1,56,56,256,type_main,content_zeros);
  udl_batch_normalization_layer(conv2_block3_3_bn_desc, &t_conv2_block3_3_bn, &t_conv2_block3_3_conv, &t_conv2_block3_3_bn_b); 

  tensor4d_t t_conv2_block3_out         = udl_tensor_create(1,56,56,256,type_main,content_zeros);
  add_desc_t conv2_block3_add_desc      = {.active = active_type_relu};
  udl_add_layer(conv2_block3_add_desc, &t_conv2_block3_out, &t_conv2_block3_3_bn, &t_conv2_block2_out);

  //block3-1

  tensor4d_t t_conv3_block1_0_conv_k       = udl_tensor_from_buffer(512,1,1,256,type_main, conv3_block1_0_conv_k);
  tensor4d_t t_conv3_block1_0_conv_b       = udl_tensor_from_buffer(1,1,2,512,type_aux, conv3_block1_0_conv_b);
  conv2d_desc_t conv3_block1_0_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 2, .sy = 2};
  tensor4d_t t_conv3_block1_0_conv         = udl_tensor_create(1,28,28,512,type_main,content_zeros);
  udl_conv2d_layer(conv3_block1_0_conv_desc, &t_conv3_block1_0_conv, &t_conv2_block3_out, &t_conv3_block1_0_conv_k, &t_conv3_block1_0_conv_b);

  tensor4d_t t_conv3_block1_0_bn_b       = udl_tensor_from_buffer(1,1,2,512,type_aux, conv3_block1_0_bn_b);
  batch_normalization_desc_t conv3_block1_0_bn_desc = {.active = active_type_linear};
  tensor4d_t t_conv3_block1_0_bn         = udl_tensor_create(1,28,28,512,type_main,content_zeros);
  udl_batch_normalization_layer(conv3_block1_0_bn_desc, &t_conv3_block1_0_bn, &t_conv3_block1_0_conv, &t_conv3_block1_0_bn_b);

  tensor4d_t t_conv3_block1_1_conv_k       = udl_tensor_from_buffer(128,1,1,256,type_main, conv3_block1_1_conv_k);
  tensor4d_t t_conv3_block1_1_conv_b       = udl_tensor_from_buffer(1,1,2,128,type_aux, conv3_block1_1_conv_b);
  conv2d_desc_t conv3_block1_1_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 2, .sy = 2};
  tensor4d_t t_conv3_block1_1_conv         = udl_tensor_create(1,28,28,128,type_main,content_zeros);
  udl_conv2d_layer(conv3_block1_1_conv_desc, &t_conv3_block1_1_conv, &t_conv2_block3_out, &t_conv3_block1_1_conv_k, &t_conv3_block1_1_conv_b);
  
  tensor4d_t t_conv3_block1_1_bn_b         = udl_tensor_from_buffer(1,1,2,128,type_aux, conv3_block1_1_bn_b);
  batch_normalization_desc_t conv3_block1_1_bn_desc = {.active = active_type_relu};
  tensor4d_t t_conv3_block1_1_relu         = udl_tensor_create(1,28,28,128,type_main,content_zeros);
  udl_batch_normalization_layer(conv3_block1_1_bn_desc, &t_conv3_block1_1_relu, &t_conv3_block1_1_conv, &t_conv3_block1_1_bn_b); 

  tensor4d_t t_conv3_block1_2_conv_k       = udl_tensor_from_buffer(128,3,3,128,type_main, conv3_block1_2_conv_k);
  tensor4d_t t_conv3_block1_2_conv_b       = udl_tensor_from_buffer(1,1,2,128,type_aux, conv3_block1_2_conv_b);
  conv2d_desc_t conv3_block1_2_conv_desc = {.active = active_type_linear, .px = 1, .py = 1, .sx = 1, .sy = 1};
  tensor4d_t t_conv3_block1_2_conv         = udl_tensor_create(1,28,28,128,type_main,content_zeros);
  udl_conv2d_layer(conv3_block1_2_conv_desc, &t_conv3_block1_2_conv, &t_conv3_block1_1_relu, &t_conv3_block1_2_conv_k, &t_conv3_block1_2_conv_b);

  tensor4d_t t_conv3_block1_2_bn_b         = udl_tensor_from_buffer(1,1,2,128,type_aux, conv3_block1_2_bn_b);
  batch_normalization_desc_t conv3_block1_2_bn_desc = {.active = active_type_relu};
  tensor4d_t t_conv3_block1_2_relu         = udl_tensor_create(1,28,28,128,type_main,content_zeros);
  udl_batch_normalization_layer(conv3_block1_2_bn_desc, &t_conv3_block1_2_relu, &t_conv3_block1_2_conv, &t_conv3_block1_2_bn_b); 

  tensor4d_t t_conv3_block1_3_conv_k       = udl_tensor_from_buffer(512,1,1,128,type_main, conv3_block1_3_conv_k);
  tensor4d_t t_conv3_block1_3_conv_b       = udl_tensor_from_buffer(1,1,2,512,type_aux, conv3_block1_3_conv_b);
  conv2d_desc_t conv3_block1_3_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 1, .sy = 1};
  tensor4d_t t_conv3_block1_3_conv         = udl_tensor_create(1,28,28,512,type_main,content_zeros);
  udl_conv2d_layer(conv3_block1_3_conv_desc, &t_conv3_block1_3_conv, &t_conv3_block1_2_relu, &t_conv3_block1_3_conv_k, &t_conv3_block1_3_conv_b);

  tensor4d_t t_conv3_block1_3_bn_b       = udl_tensor_from_buffer(1,1,2,512,type_aux, conv3_block1_3_bn_b);
  batch_normalization_desc_t conv3_block1_3_bn_desc = {.active = active_type_linear};
  tensor4d_t t_conv3_block1_3_bn         = udl_tensor_create(1,28,28,512,type_main,content_zeros);
  udl_batch_normalization_layer(conv3_block1_3_bn_desc, &t_conv3_block1_3_bn, &t_conv3_block1_3_conv, &t_conv3_block1_3_bn_b); 

  tensor4d_t t_conv3_block1_out         = udl_tensor_create(1,28,28,512,type_main,content_zeros);
  add_desc_t conv3_block1_add_desc      = {.active = active_type_relu};
  udl_add_layer(conv3_block1_add_desc, &t_conv3_block1_out, &t_conv3_block1_0_bn, &t_conv3_block1_3_bn);

  //block3-2

  tensor4d_t t_conv3_block2_1_conv_k       = udl_tensor_from_buffer(128,1,1,512,type_main, conv3_block2_1_conv_k);
  tensor4d_t t_conv3_block2_1_conv_b       = udl_tensor_from_buffer(1,1,2,128,type_aux, conv3_block2_1_conv_b);
  conv2d_desc_t conv3_block2_1_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 1, .sy = 1};
  tensor4d_t t_conv3_block2_1_conv         = udl_tensor_create(1,28,28,128,type_main,content_zeros);
  udl_conv2d_layer(conv3_block2_1_conv_desc, &t_conv3_block2_1_conv, &t_conv3_block1_out, &t_conv3_block2_1_conv_k, &t_conv3_block2_1_conv_b);
  
  tensor4d_t t_conv3_block2_1_bn_b         = udl_tensor_from_buffer(1,1,2,128,type_aux, conv3_block2_1_bn_b);
  batch_normalization_desc_t conv3_block2_1_bn_desc = {.active = active_type_relu};
  tensor4d_t t_conv3_block2_1_relu         = udl_tensor_create(1,28,28,128,type_main,content_zeros);
  udl_batch_normalization_layer(conv3_block2_1_bn_desc, &t_conv3_block2_1_relu, &t_conv3_block2_1_conv, &t_conv3_block2_1_bn_b); 

  tensor4d_t t_conv3_block2_2_conv_k       = udl_tensor_from_buffer(128,3,3,128,type_main, conv3_block2_2_conv_k);
  tensor4d_t t_conv3_block2_2_conv_b       = udl_tensor_from_buffer(1,1,2,128,type_aux, conv3_block2_2_conv_b);
  conv2d_desc_t conv3_block2_2_conv_desc = {.active = active_type_linear, .px = 1, .py = 1, .sx = 1, .sy = 1};
  tensor4d_t t_conv3_block2_2_conv         = udl_tensor_create(1,28,28,128,type_main,content_zeros);
  udl_conv2d_layer(conv3_block2_2_conv_desc, &t_conv3_block2_2_conv, &t_conv3_block2_1_relu, &t_conv3_block2_2_conv_k, &t_conv3_block2_2_conv_b);

  tensor4d_t t_conv3_block2_2_bn_b         = udl_tensor_from_buffer(1,1,2,128,type_aux, conv3_block2_2_bn_b);
  batch_normalization_desc_t conv3_block2_2_bn_desc = {.active = active_type_relu};
  tensor4d_t t_conv3_block2_2_relu         = udl_tensor_create(1,28,28,128,type_main,content_zeros);
  udl_batch_normalization_layer(conv3_block2_2_bn_desc, &t_conv3_block2_2_relu, &t_conv3_block2_2_conv, &t_conv3_block2_2_bn_b); 

  tensor4d_t t_conv3_block2_3_conv_k       = udl_tensor_from_buffer(512,1,1,128,type_main, conv3_block2_3_conv_k);
  tensor4d_t t_conv3_block2_3_conv_b       = udl_tensor_from_buffer(1,1,2,512,type_aux, conv3_block2_3_conv_b);
  conv2d_desc_t conv3_block2_3_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 1, .sy = 1};
  tensor4d_t t_conv3_block2_3_conv         = udl_tensor_create(1,28,28,512,type_main,content_zeros);
  udl_conv2d_layer(conv3_block2_3_conv_desc, &t_conv3_block2_3_conv, &t_conv3_block2_2_relu, &t_conv3_block2_3_conv_k, &t_conv3_block2_3_conv_b);

  tensor4d_t t_conv3_block2_3_bn_b       = udl_tensor_from_buffer(1,1,2,512,type_aux, conv3_block2_3_bn_b);
  batch_normalization_desc_t conv3_block2_3_bn_desc = {.active = active_type_linear};
  tensor4d_t t_conv3_block2_3_bn         = udl_tensor_create(1,28,28,512,type_main,content_zeros);
  udl_batch_normalization_layer(conv3_block2_3_bn_desc, &t_conv3_block2_3_bn, &t_conv3_block2_3_conv, &t_conv3_block2_3_bn_b); 

  tensor4d_t t_conv3_block2_out         = udl_tensor_create(1,28,28,512,type_main,content_zeros);
  add_desc_t conv3_block2_add_desc      = {.active = active_type_relu};
  udl_add_layer(conv3_block2_add_desc, &t_conv3_block2_out, &t_conv3_block1_out, &t_conv3_block2_3_bn);

  //block3-3

  tensor4d_t t_conv3_block3_1_conv_k       = udl_tensor_from_buffer(128,1,1,512,type_main, conv3_block3_1_conv_k);
  tensor4d_t t_conv3_block3_1_conv_b       = udl_tensor_from_buffer(1,1,2,128,type_aux, conv3_block3_1_conv_b);
  conv2d_desc_t conv3_block3_1_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 1, .sy = 1};
  tensor4d_t t_conv3_block3_1_conv         = udl_tensor_create(1,28,28,128,type_main,content_zeros);
  udl_conv2d_layer(conv3_block3_1_conv_desc, &t_conv3_block3_1_conv, &t_conv3_block2_out, &t_conv3_block3_1_conv_k, &t_conv3_block3_1_conv_b);
  
  tensor4d_t t_conv3_block3_1_bn_b         = udl_tensor_from_buffer(1,1,2,128,type_aux, conv3_block3_1_bn_b);
  batch_normalization_desc_t conv3_block3_1_bn_desc = {.active = active_type_relu};
  tensor4d_t t_conv3_block3_1_relu         = udl_tensor_create(1,28,28,128,type_main,content_zeros);
  udl_batch_normalization_layer(conv3_block3_1_bn_desc, &t_conv3_block3_1_relu, &t_conv3_block3_1_conv, &t_conv3_block3_1_bn_b); 

  tensor4d_t t_conv3_block3_2_conv_k       = udl_tensor_from_buffer(128,3,3,128,type_main, conv3_block3_2_conv_k);
  tensor4d_t t_conv3_block3_2_conv_b       = udl_tensor_from_buffer(1,1,2,128,type_aux, conv3_block3_2_conv_b);
  conv2d_desc_t conv3_block3_2_conv_desc = {.active = active_type_linear, .px = 1, .py = 1, .sx = 1, .sy = 1};
  tensor4d_t t_conv3_block3_2_conv         = udl_tensor_create(1,28,28,128,type_main,content_zeros);
  udl_conv2d_layer(conv3_block3_2_conv_desc, &t_conv3_block3_2_conv, &t_conv3_block3_1_relu, &t_conv3_block3_2_conv_k, &t_conv3_block3_2_conv_b);

  tensor4d_t t_conv3_block3_2_bn_b         = udl_tensor_from_buffer(1,1,2,128,type_aux, conv3_block3_2_bn_b);
  batch_normalization_desc_t conv3_block3_2_bn_desc = {.active = active_type_relu};
  tensor4d_t t_conv3_block3_2_relu         = udl_tensor_create(1,28,28,128,type_main,content_zeros);
  udl_batch_normalization_layer(conv3_block3_2_bn_desc, &t_conv3_block3_2_relu, &t_conv3_block3_2_conv, &t_conv3_block3_2_bn_b); 

  tensor4d_t t_conv3_block3_3_conv_k       = udl_tensor_from_buffer(512,1,1,128,type_main, conv3_block3_3_conv_k);
  tensor4d_t t_conv3_block3_3_conv_b       = udl_tensor_from_buffer(1,1,2,512,type_aux, conv3_block3_3_conv_b);
  conv2d_desc_t conv3_block3_3_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 1, .sy = 1};
  tensor4d_t t_conv3_block3_3_conv         = udl_tensor_create(1,28,28,512,type_main,content_zeros);
  udl_conv2d_layer(conv3_block3_3_conv_desc, &t_conv3_block3_3_conv, &t_conv3_block3_2_relu, &t_conv3_block3_3_conv_k, &t_conv3_block3_3_conv_b);

  tensor4d_t t_conv3_block3_3_bn_b       = udl_tensor_from_buffer(1,1,2,512,type_aux, conv3_block3_3_bn_b);
  batch_normalization_desc_t conv3_block3_3_bn_desc = {.active = active_type_linear};
  tensor4d_t t_conv3_block3_3_bn         = udl_tensor_create(1,28,28,512,type_main,content_zeros);
  udl_batch_normalization_layer(conv3_block3_3_bn_desc, &t_conv3_block3_3_bn, &t_conv3_block3_3_conv, &t_conv3_block3_3_bn_b); 

  tensor4d_t t_conv3_block3_out         = udl_tensor_create(1,28,28,512,type_main,content_zeros);
  add_desc_t conv3_block3_add_desc      = {.active = active_type_relu};
  udl_add_layer(conv3_block3_add_desc, &t_conv3_block3_out, &t_conv3_block2_out, &t_conv3_block3_3_bn);

  //block3-4

  tensor4d_t t_conv3_block4_1_conv_k       = udl_tensor_from_buffer(128,1,1,512,type_main, conv3_block4_1_conv_k);
  tensor4d_t t_conv3_block4_1_conv_b       = udl_tensor_from_buffer(1,1,2,128,type_aux, conv3_block4_1_conv_b);
  conv2d_desc_t conv3_block4_1_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 1, .sy = 1};
  tensor4d_t t_conv3_block4_1_conv         = udl_tensor_create(1,28,28,128,type_main,content_zeros);
  udl_conv2d_layer(conv3_block4_1_conv_desc, &t_conv3_block4_1_conv, &t_conv3_block3_out, &t_conv3_block4_1_conv_k, &t_conv3_block4_1_conv_b);
  
  tensor4d_t t_conv3_block4_1_bn_b         = udl_tensor_from_buffer(1,1,2,128,type_aux, conv3_block4_1_bn_b);
  batch_normalization_desc_t conv3_block4_1_bn_desc = {.active = active_type_relu};
  tensor4d_t t_conv3_block4_1_relu         = udl_tensor_create(1,28,28,128,type_main,content_zeros);
  udl_batch_normalization_layer(conv3_block4_1_bn_desc, &t_conv3_block4_1_relu, &t_conv3_block4_1_conv, &t_conv3_block4_1_bn_b); 

  tensor4d_t t_conv3_block4_2_conv_k       = udl_tensor_from_buffer(128,3,3,128,type_main, conv3_block4_2_conv_k);
  tensor4d_t t_conv3_block4_2_conv_b       = udl_tensor_from_buffer(1,1,2,128,type_aux, conv3_block4_2_conv_b);
  conv2d_desc_t conv3_block4_2_conv_desc = {.active = active_type_linear, .px = 1, .py = 1, .sx = 1, .sy = 1};
  tensor4d_t t_conv3_block4_2_conv         = udl_tensor_create(1,28,28,128,type_main,content_zeros);
  udl_conv2d_layer(conv3_block4_2_conv_desc, &t_conv3_block4_2_conv, &t_conv3_block4_1_relu, &t_conv3_block4_2_conv_k, &t_conv3_block4_2_conv_b);

  tensor4d_t t_conv3_block4_2_bn_b         = udl_tensor_from_buffer(1,1,2,128,type_aux, conv3_block4_2_bn_b);
  batch_normalization_desc_t conv3_block4_2_bn_desc = {.active = active_type_relu};
  tensor4d_t t_conv3_block4_2_relu         = udl_tensor_create(1,28,28,128,type_main,content_zeros);
  udl_batch_normalization_layer(conv3_block4_2_bn_desc, &t_conv3_block4_2_relu, &t_conv3_block4_2_conv, &t_conv3_block4_2_bn_b); 

  tensor4d_t t_conv3_block4_3_conv_k       = udl_tensor_from_buffer(512,1,1,128,type_main, conv3_block4_3_conv_k);
  tensor4d_t t_conv3_block4_3_conv_b       = udl_tensor_from_buffer(1,1,2,512,type_aux, conv3_block4_3_conv_b);
  conv2d_desc_t conv3_block4_3_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 1, .sy = 1};
  tensor4d_t t_conv3_block4_3_conv         = udl_tensor_create(1,28,28,512,type_main,content_zeros);
  udl_conv2d_layer(conv3_block4_3_conv_desc, &t_conv3_block4_3_conv, &t_conv3_block4_2_relu, &t_conv3_block4_3_conv_k, &t_conv3_block4_3_conv_b);

  tensor4d_t t_conv3_block4_3_bn_b       = udl_tensor_from_buffer(1,1,2,512,type_aux, conv3_block4_3_bn_b);
  batch_normalization_desc_t conv3_block4_3_bn_desc = {.active = active_type_linear};
  tensor4d_t t_conv3_block4_3_bn         = udl_tensor_create(1,28,28,512,type_main,content_zeros);
  udl_batch_normalization_layer(conv3_block4_3_bn_desc, &t_conv3_block4_3_bn, &t_conv3_block4_3_conv, &t_conv3_block4_3_bn_b); 

  tensor4d_t t_conv3_block4_out         = udl_tensor_create(1,28,28,512,type_main,content_zeros);
  add_desc_t conv3_block4_add_desc      = {.active = active_type_relu};
  udl_add_layer(conv3_block4_add_desc, &t_conv3_block4_out, &t_conv3_block3_out, &t_conv3_block4_3_bn);

  //block4-1

  tensor4d_t t_conv4_block1_0_conv_k       = udl_tensor_from_buffer(1024,1,1,512,type_main, conv4_block1_0_conv_k);
  tensor4d_t t_conv4_block1_0_conv_b       = udl_tensor_from_buffer(1,1,2,1024,type_aux, conv4_block1_0_conv_b);
  conv2d_desc_t conv4_block1_0_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 2, .sy = 2};
  tensor4d_t t_conv4_block1_0_conv         = udl_tensor_create(1,14,14,1024,type_main,content_zeros);
  udl_conv2d_layer(conv4_block1_0_conv_desc, &t_conv4_block1_0_conv, &t_conv3_block4_out, &t_conv4_block1_0_conv_k, &t_conv4_block1_0_conv_b);

  tensor4d_t t_conv4_block1_0_bn_b       = udl_tensor_from_buffer(1,1,2,1024,type_aux, conv4_block1_0_bn_b);
  batch_normalization_desc_t conv4_block1_0_bn_desc = {.active = active_type_linear};
  tensor4d_t t_conv4_block1_0_bn         = udl_tensor_create(1,14,14,1024,type_main,content_zeros);
  udl_batch_normalization_layer(conv4_block1_0_bn_desc, &t_conv4_block1_0_bn, &t_conv4_block1_0_conv, &t_conv4_block1_0_bn_b);

  tensor4d_t t_conv4_block1_1_conv_k       = udl_tensor_from_buffer(256,1,1,512,type_main, conv4_block1_1_conv_k);
  tensor4d_t t_conv4_block1_1_conv_b       = udl_tensor_from_buffer(1,1,2,256,type_aux, conv4_block1_1_conv_b);
  conv2d_desc_t conv4_block1_1_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 2, .sy = 2};
  tensor4d_t t_conv4_block1_1_conv         = udl_tensor_create(1,14,14,256,type_main,content_zeros);
  udl_conv2d_layer(conv4_block1_1_conv_desc, &t_conv4_block1_1_conv, &t_conv3_block4_out, &t_conv4_block1_1_conv_k, &t_conv4_block1_1_conv_b);
  
  tensor4d_t t_conv4_block1_1_bn_b         = udl_tensor_from_buffer(1,1,2,256,type_aux, conv4_block1_1_bn_b);
  batch_normalization_desc_t conv4_block1_1_bn_desc = {.active = active_type_relu};
  tensor4d_t t_conv4_block1_1_relu         = udl_tensor_create(1,14,14,256,type_main,content_zeros);
  udl_batch_normalization_layer(conv4_block1_1_bn_desc, &t_conv4_block1_1_relu, &t_conv4_block1_1_conv, &t_conv4_block1_1_bn_b); 

  tensor4d_t t_conv4_block1_2_conv_k       = udl_tensor_from_buffer(256,3,3,256,type_main, conv4_block1_2_conv_k);
  tensor4d_t t_conv4_block1_2_conv_b       = udl_tensor_from_buffer(1,1,2,256,type_aux, conv4_block1_2_conv_b);
  conv2d_desc_t conv4_block1_2_conv_desc = {.active = active_type_linear, .px = 1, .py = 1, .sx = 1, .sy = 1};
  tensor4d_t t_conv4_block1_2_conv         = udl_tensor_create(1,14,14,256,type_main,content_zeros);
  udl_conv2d_layer(conv4_block1_2_conv_desc, &t_conv4_block1_2_conv, &t_conv4_block1_1_relu, &t_conv4_block1_2_conv_k, &t_conv4_block1_2_conv_b);

  tensor4d_t t_conv4_block1_2_bn_b         = udl_tensor_from_buffer(1,1,2,256,type_aux, conv4_block1_2_bn_b);
  batch_normalization_desc_t conv4_block1_2_bn_desc = {.active = active_type_relu};
  tensor4d_t t_conv4_block1_2_relu         = udl_tensor_create(1,14,14,256,type_main,content_zeros);
  udl_batch_normalization_layer(conv4_block1_2_bn_desc, &t_conv4_block1_2_relu, &t_conv4_block1_2_conv, &t_conv4_block1_2_bn_b); 

  tensor4d_t t_conv4_block1_3_conv_k       = udl_tensor_from_buffer(1024,1,1,256,type_main, conv4_block1_3_conv_k);
  tensor4d_t t_conv4_block1_3_conv_b       = udl_tensor_from_buffer(1,1,2,1024,type_aux, conv4_block1_3_conv_b);
  conv2d_desc_t conv4_block1_3_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 1, .sy = 1};
  tensor4d_t t_conv4_block1_3_conv         = udl_tensor_create(1,14,14,1024,type_main,content_zeros);
  udl_conv2d_layer(conv4_block1_3_conv_desc, &t_conv4_block1_3_conv, &t_conv4_block1_2_relu, &t_conv4_block1_3_conv_k, &t_conv4_block1_3_conv_b);

  tensor4d_t t_conv4_block1_3_bn_b       = udl_tensor_from_buffer(1,1,2,1024,type_aux, conv4_block1_3_bn_b);
  batch_normalization_desc_t conv4_block1_3_bn_desc = {.active = active_type_linear};
  tensor4d_t t_conv4_block1_3_bn         = udl_tensor_create(1,14,14,1024,type_main,content_zeros);
  udl_batch_normalization_layer(conv4_block1_3_bn_desc, &t_conv4_block1_3_bn, &t_conv4_block1_3_conv, &t_conv4_block1_3_bn_b); 

  tensor4d_t t_conv4_block1_out         = udl_tensor_create(1,14,14,1024,type_main,content_zeros);
  add_desc_t conv4_block1_add_desc      = {.active = active_type_relu};
  udl_add_layer(conv4_block1_add_desc, &t_conv4_block1_out, &t_conv4_block1_0_bn, &t_conv4_block1_3_bn);

  //block4-2
  tensor4d_t t_conv4_block2_1_conv_k       = udl_tensor_from_buffer(256,1,1,1024,type_main, conv4_block2_1_conv_k);
  tensor4d_t t_conv4_block2_1_conv_b       = udl_tensor_from_buffer(1,1,2,256,type_aux, conv4_block2_1_conv_b);
  conv2d_desc_t conv4_block2_1_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 1, .sy = 1};
  tensor4d_t t_conv4_block2_1_conv         = udl_tensor_create(1,14,14,256,type_main,content_zeros);
  udl_conv2d_layer(conv4_block2_1_conv_desc, &t_conv4_block2_1_conv, &t_conv4_block1_out, &t_conv4_block2_1_conv_k, &t_conv4_block2_1_conv_b);
  
  tensor4d_t t_conv4_block2_1_bn_b         = udl_tensor_from_buffer(1,1,2,256,type_aux, conv4_block2_1_bn_b);
  batch_normalization_desc_t conv4_block2_1_bn_desc = {.active = active_type_relu};
  tensor4d_t t_conv4_block2_1_relu         = udl_tensor_create(1,14,14,256,type_main,content_zeros);
  udl_batch_normalization_layer(conv4_block2_1_bn_desc, &t_conv4_block2_1_relu, &t_conv4_block2_1_conv, &t_conv4_block2_1_bn_b); 

  tensor4d_t t_conv4_block2_2_conv_k       = udl_tensor_from_buffer(256,3,3,256,type_main, conv4_block2_2_conv_k);
  tensor4d_t t_conv4_block2_2_conv_b       = udl_tensor_from_buffer(1,1,2,256,type_aux, conv4_block2_2_conv_b);
  conv2d_desc_t conv4_block2_2_conv_desc = {.active = active_type_linear, .px = 1, .py = 1, .sx = 1, .sy = 1};
  tensor4d_t t_conv4_block2_2_conv         = udl_tensor_create(1,14,14,256,type_main,content_zeros);
  udl_conv2d_layer(conv4_block2_2_conv_desc, &t_conv4_block2_2_conv, &t_conv4_block2_1_relu, &t_conv4_block2_2_conv_k, &t_conv4_block2_2_conv_b);

  tensor4d_t t_conv4_block2_2_bn_b         = udl_tensor_from_buffer(1,1,2,256,type_aux, conv4_block2_2_bn_b);
  batch_normalization_desc_t conv4_block2_2_bn_desc = {.active = active_type_relu};
  tensor4d_t t_conv4_block2_2_relu         = udl_tensor_create(1,14,14,256,type_main,content_zeros);
  udl_batch_normalization_layer(conv4_block2_2_bn_desc, &t_conv4_block2_2_relu, &t_conv4_block2_2_conv, &t_conv4_block2_2_bn_b); 

  tensor4d_t t_conv4_block2_3_conv_k       = udl_tensor_from_buffer(1024,1,1,256,type_main, conv4_block2_3_conv_k);
  tensor4d_t t_conv4_block2_3_conv_b       = udl_tensor_from_buffer(1,1,2,1024,type_aux, conv4_block2_3_conv_b);
  conv2d_desc_t conv4_block2_3_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 1, .sy = 1};
  tensor4d_t t_conv4_block2_3_conv         = udl_tensor_create(1,14,14,1024,type_main,content_zeros);
  udl_conv2d_layer(conv4_block2_3_conv_desc, &t_conv4_block2_3_conv, &t_conv4_block2_2_relu, &t_conv4_block2_3_conv_k, &t_conv4_block2_3_conv_b);

  tensor4d_t t_conv4_block2_3_bn_b       = udl_tensor_from_buffer(1,1,2,1024,type_aux, conv4_block2_3_bn_b);
  batch_normalization_desc_t conv4_block2_3_bn_desc = {.active = active_type_linear};
  tensor4d_t t_conv4_block2_3_bn         = udl_tensor_create(1,14,14,1024,type_main,content_zeros);
  udl_batch_normalization_layer(conv4_block2_3_bn_desc, &t_conv4_block2_3_bn, &t_conv4_block2_3_conv, &t_conv4_block2_3_bn_b); 

  tensor4d_t t_conv4_block2_out         = udl_tensor_create(1,14,14,1024,type_main,content_zeros);
  add_desc_t conv4_block2_add_desc      = {.active = active_type_relu};
  udl_add_layer(conv4_block2_add_desc, &t_conv4_block2_out, &t_conv4_block1_out, &t_conv4_block2_3_bn);

  //block4-3
  tensor4d_t t_conv4_block3_1_conv_k       = udl_tensor_from_buffer(256,1,1,1024,type_main, conv4_block3_1_conv_k);
  tensor4d_t t_conv4_block3_1_conv_b       = udl_tensor_from_buffer(1,1,2,256,type_aux, conv4_block3_1_conv_b);
  conv2d_desc_t conv4_block3_1_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 1, .sy = 1};
  tensor4d_t t_conv4_block3_1_conv         = udl_tensor_create(1,14,14,256,type_main,content_zeros);
  udl_conv2d_layer(conv4_block3_1_conv_desc, &t_conv4_block3_1_conv, &t_conv4_block2_out, &t_conv4_block3_1_conv_k, &t_conv4_block3_1_conv_b);
  
  tensor4d_t t_conv4_block3_1_bn_b         = udl_tensor_from_buffer(1,1,2,256,type_aux, conv4_block3_1_bn_b);
  batch_normalization_desc_t conv4_block3_1_bn_desc = {.active = active_type_relu};
  tensor4d_t t_conv4_block3_1_relu         = udl_tensor_create(1,14,14,256,type_main,content_zeros);
  udl_batch_normalization_layer(conv4_block3_1_bn_desc, &t_conv4_block3_1_relu, &t_conv4_block3_1_conv, &t_conv4_block3_1_bn_b); 

  tensor4d_t t_conv4_block3_2_conv_k       = udl_tensor_from_buffer(256,3,3,256,type_main, conv4_block3_2_conv_k);
  tensor4d_t t_conv4_block3_2_conv_b       = udl_tensor_from_buffer(1,1,2,256,type_aux, conv4_block3_2_conv_b);
  conv2d_desc_t conv4_block3_2_conv_desc = {.active = active_type_linear, .px = 1, .py = 1, .sx = 1, .sy = 1};
  tensor4d_t t_conv4_block3_2_conv         = udl_tensor_create(1,14,14,256,type_main,content_zeros);
  udl_conv2d_layer(conv4_block3_2_conv_desc, &t_conv4_block3_2_conv, &t_conv4_block3_1_relu, &t_conv4_block3_2_conv_k, &t_conv4_block3_2_conv_b);

  tensor4d_t t_conv4_block3_2_bn_b         = udl_tensor_from_buffer(1,1,2,256,type_aux, conv4_block3_2_bn_b);
  batch_normalization_desc_t conv4_block3_2_bn_desc = {.active = active_type_relu};
  tensor4d_t t_conv4_block3_2_relu         = udl_tensor_create(1,14,14,256,type_main,content_zeros);
  udl_batch_normalization_layer(conv4_block3_2_bn_desc, &t_conv4_block3_2_relu, &t_conv4_block3_2_conv, &t_conv4_block3_2_bn_b); 

  tensor4d_t t_conv4_block3_3_conv_k       = udl_tensor_from_buffer(1024,1,1,256,type_main, conv4_block3_3_conv_k);
  tensor4d_t t_conv4_block3_3_conv_b       = udl_tensor_from_buffer(1,1,2,1024,type_aux, conv4_block3_3_conv_b);
  conv2d_desc_t conv4_block3_3_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 1, .sy = 1};
  tensor4d_t t_conv4_block3_3_conv         = udl_tensor_create(1,14,14,1024,type_main,content_zeros);
  udl_conv2d_layer(conv4_block3_3_conv_desc, &t_conv4_block3_3_conv, &t_conv4_block3_2_relu, &t_conv4_block3_3_conv_k, &t_conv4_block3_3_conv_b);

  tensor4d_t t_conv4_block3_3_bn_b       = udl_tensor_from_buffer(1,1,2,1024,type_aux, conv4_block3_3_bn_b);
  batch_normalization_desc_t conv4_block3_3_bn_desc = {.active = active_type_linear};
  tensor4d_t t_conv4_block3_3_bn         = udl_tensor_create(1,14,14,1024,type_main,content_zeros);
  udl_batch_normalization_layer(conv4_block3_3_bn_desc, &t_conv4_block3_3_bn, &t_conv4_block3_3_conv, &t_conv4_block3_3_bn_b); 

  tensor4d_t t_conv4_block3_out         = udl_tensor_create(1,14,14,1024,type_main,content_zeros);
  add_desc_t conv4_block3_add_desc      = {.active = active_type_relu};
  udl_add_layer(conv4_block3_add_desc, &t_conv4_block3_out, &t_conv4_block2_out, &t_conv4_block3_3_bn);

  //block4-4
  tensor4d_t t_conv4_block4_1_conv_k       = udl_tensor_from_buffer(256,1,1,1024,type_main, conv4_block4_1_conv_k);
  tensor4d_t t_conv4_block4_1_conv_b       = udl_tensor_from_buffer(1,1,2,256,type_aux, conv4_block4_1_conv_b);
  conv2d_desc_t conv4_block4_1_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 1, .sy = 1};
  tensor4d_t t_conv4_block4_1_conv         = udl_tensor_create(1,14,14,256,type_main,content_zeros);
  udl_conv2d_layer(conv4_block4_1_conv_desc, &t_conv4_block4_1_conv, &t_conv4_block3_out, &t_conv4_block4_1_conv_k, &t_conv4_block4_1_conv_b);
  
  tensor4d_t t_conv4_block4_1_bn_b         = udl_tensor_from_buffer(1,1,2,256,type_aux, conv4_block4_1_bn_b);
  batch_normalization_desc_t conv4_block4_1_bn_desc = {.active = active_type_relu};
  tensor4d_t t_conv4_block4_1_relu         = udl_tensor_create(1,14,14,256,type_main,content_zeros);
  udl_batch_normalization_layer(conv4_block4_1_bn_desc, &t_conv4_block4_1_relu, &t_conv4_block4_1_conv, &t_conv4_block4_1_bn_b); 

  tensor4d_t t_conv4_block4_2_conv_k       = udl_tensor_from_buffer(256,3,3,256,type_main, conv4_block4_2_conv_k);
  tensor4d_t t_conv4_block4_2_conv_b       = udl_tensor_from_buffer(1,1,2,256,type_aux, conv4_block4_2_conv_b);
  conv2d_desc_t conv4_block4_2_conv_desc = {.active = active_type_linear, .px = 1, .py = 1, .sx = 1, .sy = 1};
  tensor4d_t t_conv4_block4_2_conv         = udl_tensor_create(1,14,14,256,type_main,content_zeros);
  udl_conv2d_layer(conv4_block4_2_conv_desc, &t_conv4_block4_2_conv, &t_conv4_block4_1_relu, &t_conv4_block4_2_conv_k, &t_conv4_block4_2_conv_b);

  tensor4d_t t_conv4_block4_2_bn_b         = udl_tensor_from_buffer(1,1,2,256,type_aux, conv4_block4_2_bn_b);
  batch_normalization_desc_t conv4_block4_2_bn_desc = {.active = active_type_relu};
  tensor4d_t t_conv4_block4_2_relu         = udl_tensor_create(1,14,14,256,type_main,content_zeros);
  udl_batch_normalization_layer(conv4_block4_2_bn_desc, &t_conv4_block4_2_relu, &t_conv4_block4_2_conv, &t_conv4_block4_2_bn_b); 

  tensor4d_t t_conv4_block4_3_conv_k       = udl_tensor_from_buffer(1024,1,1,256,type_main, conv4_block4_3_conv_k);
  tensor4d_t t_conv4_block4_3_conv_b       = udl_tensor_from_buffer(1,1,2,1024,type_aux, conv4_block4_3_conv_b);
  conv2d_desc_t conv4_block4_3_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 1, .sy = 1};
  tensor4d_t t_conv4_block4_3_conv         = udl_tensor_create(1,14,14,1024,type_main,content_zeros);
  udl_conv2d_layer(conv4_block4_3_conv_desc, &t_conv4_block4_3_conv, &t_conv4_block4_2_relu, &t_conv4_block4_3_conv_k, &t_conv4_block4_3_conv_b);

  tensor4d_t t_conv4_block4_3_bn_b       = udl_tensor_from_buffer(1,1,2,1024,type_aux, conv4_block4_3_bn_b);
  batch_normalization_desc_t conv4_block4_3_bn_desc = {.active = active_type_linear};
  tensor4d_t t_conv4_block4_3_bn         = udl_tensor_create(1,14,14,1024,type_main,content_zeros);
  udl_batch_normalization_layer(conv4_block4_3_bn_desc, &t_conv4_block4_3_bn, &t_conv4_block4_3_conv, &t_conv4_block4_3_bn_b); 

  tensor4d_t t_conv4_block4_out         = udl_tensor_create(1,14,14,1024,type_main,content_zeros);
  add_desc_t conv4_block4_add_desc      = {.active = active_type_relu};
  udl_add_layer(conv4_block4_add_desc, &t_conv4_block4_out, &t_conv4_block3_out, &t_conv4_block4_3_bn);

  //block4-5
  tensor4d_t t_conv4_block5_1_conv_k       = udl_tensor_from_buffer(256,1,1,1024,type_main, conv4_block5_1_conv_k);
  tensor4d_t t_conv4_block5_1_conv_b       = udl_tensor_from_buffer(1,1,2,256,type_aux, conv4_block5_1_conv_b);
  conv2d_desc_t conv4_block5_1_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 1, .sy = 1};
  tensor4d_t t_conv4_block5_1_conv         = udl_tensor_create(1,14,14,256,type_main,content_zeros);
  udl_conv2d_layer(conv4_block5_1_conv_desc, &t_conv4_block5_1_conv, &t_conv4_block4_out, &t_conv4_block5_1_conv_k, &t_conv4_block5_1_conv_b);
  
  tensor4d_t t_conv4_block5_1_bn_b         = udl_tensor_from_buffer(1,1,2,256,type_aux, conv4_block5_1_bn_b);
  batch_normalization_desc_t conv4_block5_1_bn_desc = {.active = active_type_relu};
  tensor4d_t t_conv4_block5_1_relu         = udl_tensor_create(1,14,14,256,type_main,content_zeros);
  udl_batch_normalization_layer(conv4_block5_1_bn_desc, &t_conv4_block5_1_relu, &t_conv4_block5_1_conv, &t_conv4_block5_1_bn_b); 

  tensor4d_t t_conv4_block5_2_conv_k       = udl_tensor_from_buffer(256,3,3,256,type_main, conv4_block5_2_conv_k);
  tensor4d_t t_conv4_block5_2_conv_b       = udl_tensor_from_buffer(1,1,2,256,type_aux, conv4_block5_2_conv_b);
  conv2d_desc_t conv4_block5_2_conv_desc = {.active = active_type_linear, .px = 1, .py = 1, .sx = 1, .sy = 1};
  tensor4d_t t_conv4_block5_2_conv         = udl_tensor_create(1,14,14,256,type_main,content_zeros);
  udl_conv2d_layer(conv4_block5_2_conv_desc, &t_conv4_block5_2_conv, &t_conv4_block5_1_relu, &t_conv4_block5_2_conv_k, &t_conv4_block5_2_conv_b);

  tensor4d_t t_conv4_block5_2_bn_b         = udl_tensor_from_buffer(1,1,2,256,type_aux, conv4_block5_2_bn_b);
  batch_normalization_desc_t conv4_block5_2_bn_desc = {.active = active_type_relu};
  tensor4d_t t_conv4_block5_2_relu         = udl_tensor_create(1,14,14,256,type_main,content_zeros);
  udl_batch_normalization_layer(conv4_block5_2_bn_desc, &t_conv4_block5_2_relu, &t_conv4_block5_2_conv, &t_conv4_block5_2_bn_b); 

  tensor4d_t t_conv4_block5_3_conv_k       = udl_tensor_from_buffer(1024,1,1,256,type_main, conv4_block5_3_conv_k);
  tensor4d_t t_conv4_block5_3_conv_b       = udl_tensor_from_buffer(1,1,2,1024,type_aux, conv4_block5_3_conv_b);
  conv2d_desc_t conv4_block5_3_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 1, .sy = 1};
  tensor4d_t t_conv4_block5_3_conv         = udl_tensor_create(1,14,14,1024,type_main,content_zeros);
  udl_conv2d_layer(conv4_block5_3_conv_desc, &t_conv4_block5_3_conv, &t_conv4_block5_2_relu, &t_conv4_block5_3_conv_k, &t_conv4_block5_3_conv_b);

  tensor4d_t t_conv4_block5_3_bn_b       = udl_tensor_from_buffer(1,1,2,1024,type_aux, conv4_block5_3_bn_b);
  batch_normalization_desc_t conv4_block5_3_bn_desc = {.active = active_type_linear};
  tensor4d_t t_conv4_block5_3_bn         = udl_tensor_create(1,14,14,1024,type_main,content_zeros);
  udl_batch_normalization_layer(conv4_block5_3_bn_desc, &t_conv4_block5_3_bn, &t_conv4_block5_3_conv, &t_conv4_block5_3_bn_b); 

  tensor4d_t t_conv4_block5_out         = udl_tensor_create(1,14,14,1024,type_main,content_zeros);
  add_desc_t conv4_block5_add_desc      = {.active = active_type_relu};
  udl_add_layer(conv4_block5_add_desc, &t_conv4_block5_out, &t_conv4_block4_out, &t_conv4_block5_3_bn);

  //block4-6
  tensor4d_t t_conv4_block6_1_conv_k       = udl_tensor_from_buffer(256,1,1,1024,type_main, conv4_block6_1_conv_k);
  tensor4d_t t_conv4_block6_1_conv_b       = udl_tensor_from_buffer(1,1,2,256,type_aux, conv4_block6_1_conv_b);
  conv2d_desc_t conv4_block6_1_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 1, .sy = 1};
  tensor4d_t t_conv4_block6_1_conv         = udl_tensor_create(1,14,14,256,type_main,content_zeros);
  udl_conv2d_layer(conv4_block6_1_conv_desc, &t_conv4_block6_1_conv, &t_conv4_block5_out, &t_conv4_block6_1_conv_k, &t_conv4_block6_1_conv_b);
  
  tensor4d_t t_conv4_block6_1_bn_b         = udl_tensor_from_buffer(1,1,2,256,type_aux, conv4_block6_1_bn_b);
  batch_normalization_desc_t conv4_block6_1_bn_desc = {.active = active_type_relu};
  tensor4d_t t_conv4_block6_1_relu         = udl_tensor_create(1,14,14,256,type_main,content_zeros);
  udl_batch_normalization_layer(conv4_block6_1_bn_desc, &t_conv4_block6_1_relu, &t_conv4_block6_1_conv, &t_conv4_block6_1_bn_b); 

  tensor4d_t t_conv4_block6_2_conv_k       = udl_tensor_from_buffer(256,3,3,256,type_main, conv4_block6_2_conv_k);
  tensor4d_t t_conv4_block6_2_conv_b       = udl_tensor_from_buffer(1,1,2,256,type_aux, conv4_block6_2_conv_b);
  conv2d_desc_t conv4_block6_2_conv_desc = {.active = active_type_linear, .px = 1, .py = 1, .sx = 1, .sy = 1};
  tensor4d_t t_conv4_block6_2_conv         = udl_tensor_create(1,14,14,256,type_main,content_zeros);
  udl_conv2d_layer(conv4_block6_2_conv_desc, &t_conv4_block6_2_conv, &t_conv4_block6_1_relu, &t_conv4_block6_2_conv_k, &t_conv4_block6_2_conv_b);

  tensor4d_t t_conv4_block6_2_bn_b         = udl_tensor_from_buffer(1,1,2,256,type_aux, conv4_block6_2_bn_b);
  batch_normalization_desc_t conv4_block6_2_bn_desc = {.active = active_type_relu};
  tensor4d_t t_conv4_block6_2_relu         = udl_tensor_create(1,14,14,256,type_main,content_zeros);
  udl_batch_normalization_layer(conv4_block6_2_bn_desc, &t_conv4_block6_2_relu, &t_conv4_block6_2_conv, &t_conv4_block6_2_bn_b); 

  tensor4d_t t_conv4_block6_3_conv_k       = udl_tensor_from_buffer(1024,1,1,256,type_main, conv4_block6_3_conv_k);
  tensor4d_t t_conv4_block6_3_conv_b       = udl_tensor_from_buffer(1,1,2,1024,type_aux, conv4_block6_3_conv_b);
  conv2d_desc_t conv4_block6_3_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 1, .sy = 1};
  tensor4d_t t_conv4_block6_3_conv         = udl_tensor_create(1,14,14,1024,type_main,content_zeros);
  udl_conv2d_layer(conv4_block6_3_conv_desc, &t_conv4_block6_3_conv, &t_conv4_block6_2_relu, &t_conv4_block6_3_conv_k, &t_conv4_block6_3_conv_b);

  tensor4d_t t_conv4_block6_3_bn_b       = udl_tensor_from_buffer(1,1,2,1024,type_aux, conv4_block6_3_bn_b);
  batch_normalization_desc_t conv4_block6_3_bn_desc = {.active = active_type_linear};
  tensor4d_t t_conv4_block6_3_bn         = udl_tensor_create(1,14,14,1024,type_main,content_zeros);
  udl_batch_normalization_layer(conv4_block6_3_bn_desc, &t_conv4_block6_3_bn, &t_conv4_block6_3_conv, &t_conv4_block6_3_bn_b); 

  tensor4d_t t_conv4_block6_out         = udl_tensor_create(1,14,14,1024,type_main,content_zeros);
  add_desc_t conv4_block6_add_desc      = {.active = active_type_relu};
  udl_add_layer(conv4_block6_add_desc, &t_conv4_block6_out, &t_conv4_block5_out, &t_conv4_block6_3_bn);

  //block5-1

  tensor4d_t t_conv5_block1_0_conv_k       = udl_tensor_from_buffer(2048,1,1,1024,type_main, conv5_block1_0_conv_k);
  tensor4d_t t_conv5_block1_0_conv_b       = udl_tensor_from_buffer(1,1,2,2048,type_aux, conv5_block1_0_conv_b);
  conv2d_desc_t conv5_block1_0_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 2, .sy = 2};
  tensor4d_t t_conv5_block1_0_conv         = udl_tensor_create(1,7,7,2048,type_main,content_zeros);
  udl_conv2d_layer(conv5_block1_0_conv_desc, &t_conv5_block1_0_conv, &t_conv4_block6_out, &t_conv5_block1_0_conv_k, &t_conv5_block1_0_conv_b);

  tensor4d_t t_conv5_block1_0_bn_b       = udl_tensor_from_buffer(1,1,2,2048,type_aux, conv5_block1_0_bn_b);
  batch_normalization_desc_t conv5_block1_0_bn_desc = {.active = active_type_linear};
  tensor4d_t t_conv5_block1_0_bn         = udl_tensor_create(1,7,7,2048,type_main,content_zeros);
  udl_batch_normalization_layer(conv5_block1_0_bn_desc, &t_conv5_block1_0_bn, &t_conv5_block1_0_conv, &t_conv5_block1_0_bn_b);

  tensor4d_t t_conv5_block1_1_conv_k       = udl_tensor_from_buffer(512,1,1,1024,type_main, conv5_block1_1_conv_k);
  tensor4d_t t_conv5_block1_1_conv_b       = udl_tensor_from_buffer(1,1,2,512,type_aux, conv5_block1_1_conv_b);
  conv2d_desc_t conv5_block1_1_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 2, .sy = 2};
  tensor4d_t t_conv5_block1_1_conv         = udl_tensor_create(1,7,7,512,type_main,content_zeros);
  udl_conv2d_layer(conv5_block1_1_conv_desc, &t_conv5_block1_1_conv, &t_conv4_block6_out, &t_conv5_block1_1_conv_k, &t_conv5_block1_1_conv_b);
  
  tensor4d_t t_conv5_block1_1_bn_b         = udl_tensor_from_buffer(1,1,2,512,type_aux, conv5_block1_1_bn_b);
  batch_normalization_desc_t conv5_block1_1_bn_desc = {.active = active_type_relu};
  tensor4d_t t_conv5_block1_1_relu         = udl_tensor_create(1,7,7,512,type_main,content_zeros);
  udl_batch_normalization_layer(conv5_block1_1_bn_desc, &t_conv5_block1_1_relu, &t_conv5_block1_1_conv, &t_conv5_block1_1_bn_b); 

  tensor4d_t t_conv5_block1_2_conv_k       = udl_tensor_from_buffer(512,3,3,512,type_main, conv5_block1_2_conv_k);
  tensor4d_t t_conv5_block1_2_conv_b       = udl_tensor_from_buffer(1,1,2,512,type_aux, conv5_block1_2_conv_b);
  conv2d_desc_t conv5_block1_2_conv_desc = {.active = active_type_linear, .px = 1, .py = 1, .sx = 1, .sy = 1};
  tensor4d_t t_conv5_block1_2_conv         = udl_tensor_create(1,7,7,512,type_main,content_zeros);
  udl_conv2d_layer(conv5_block1_2_conv_desc, &t_conv5_block1_2_conv, &t_conv5_block1_1_relu, &t_conv5_block1_2_conv_k, &t_conv5_block1_2_conv_b);

  tensor4d_t t_conv5_block1_2_bn_b         = udl_tensor_from_buffer(1,1,2,512,type_aux, conv5_block1_2_bn_b);
  batch_normalization_desc_t conv5_block1_2_bn_desc = {.active = active_type_relu};
  tensor4d_t t_conv5_block1_2_relu         = udl_tensor_create(1,7,7,512,type_main,content_zeros);
  udl_batch_normalization_layer(conv5_block1_2_bn_desc, &t_conv5_block1_2_relu, &t_conv5_block1_2_conv, &t_conv5_block1_2_bn_b); 

  tensor4d_t t_conv5_block1_3_conv_k       = udl_tensor_from_buffer(2048,1,1,512,type_main, conv5_block1_3_conv_k);
  tensor4d_t t_conv5_block1_3_conv_b       = udl_tensor_from_buffer(1,1,2,2048,type_aux, conv5_block1_3_conv_b);
  conv2d_desc_t conv5_block1_3_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 1, .sy = 1};
  tensor4d_t t_conv5_block1_3_conv         = udl_tensor_create(1,7,7,2048,type_main,content_zeros);
  udl_conv2d_layer(conv5_block1_3_conv_desc, &t_conv5_block1_3_conv, &t_conv5_block1_2_relu, &t_conv5_block1_3_conv_k, &t_conv5_block1_3_conv_b);

  tensor4d_t t_conv5_block1_3_bn_b       = udl_tensor_from_buffer(1,1,2,2048,type_aux, conv5_block1_3_bn_b);
  batch_normalization_desc_t conv5_block1_3_bn_desc = {.active = active_type_linear};
  tensor4d_t t_conv5_block1_3_bn         = udl_tensor_create(1,7,7,2048,type_main,content_zeros);
  udl_batch_normalization_layer(conv5_block1_3_bn_desc, &t_conv5_block1_3_bn, &t_conv5_block1_3_conv, &t_conv5_block1_3_bn_b); 

  tensor4d_t t_conv5_block1_out         = udl_tensor_create(1,7,7,2048,type_main,content_zeros);
  add_desc_t conv5_block1_add_desc      = {.active = active_type_relu};
  udl_add_layer(conv5_block1_add_desc, &t_conv5_block1_out, &t_conv5_block1_0_bn, &t_conv5_block1_3_bn);

  //block5-2
  tensor4d_t t_conv5_block2_1_conv_k       = udl_tensor_from_buffer(512,1,1,2048,type_main, conv5_block2_1_conv_k);
  tensor4d_t t_conv5_block2_1_conv_b       = udl_tensor_from_buffer(1,1,2,512,type_aux, conv5_block2_1_conv_b);
  conv2d_desc_t conv5_block2_1_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 1, .sy = 1};
  tensor4d_t t_conv5_block2_1_conv         = udl_tensor_create(1,7,7,512,type_main,content_zeros);
  udl_conv2d_layer(conv5_block2_1_conv_desc, &t_conv5_block2_1_conv, &t_conv5_block1_out, &t_conv5_block2_1_conv_k, &t_conv5_block2_1_conv_b);
  
  tensor4d_t t_conv5_block2_1_bn_b         = udl_tensor_from_buffer(1,1,2,512,type_aux, conv5_block2_1_bn_b);
  batch_normalization_desc_t conv5_block2_1_bn_desc = {.active = active_type_relu};
  tensor4d_t t_conv5_block2_1_relu         = udl_tensor_create(1,7,7,512,type_main,content_zeros);
  udl_batch_normalization_layer(conv5_block2_1_bn_desc, &t_conv5_block2_1_relu, &t_conv5_block2_1_conv, &t_conv5_block2_1_bn_b); 

  tensor4d_t t_conv5_block2_2_conv_k       = udl_tensor_from_buffer(512,3,3,512,type_main, conv5_block2_2_conv_k);
  tensor4d_t t_conv5_block2_2_conv_b       = udl_tensor_from_buffer(1,1,2,512,type_aux, conv5_block2_2_conv_b);
  conv2d_desc_t conv5_block2_2_conv_desc = {.active = active_type_linear, .px = 1, .py = 1, .sx = 1, .sy = 1};
  tensor4d_t t_conv5_block2_2_conv         = udl_tensor_create(1,7,7,512,type_main,content_zeros);
  udl_conv2d_layer(conv5_block2_2_conv_desc, &t_conv5_block2_2_conv, &t_conv5_block2_1_relu, &t_conv5_block2_2_conv_k, &t_conv5_block2_2_conv_b);

  tensor4d_t t_conv5_block2_2_bn_b         = udl_tensor_from_buffer(1,1,2,512,type_aux, conv5_block2_2_bn_b);
  batch_normalization_desc_t conv5_block2_2_bn_desc = {.active = active_type_relu};
  tensor4d_t t_conv5_block2_2_relu         = udl_tensor_create(1,7,7,512,type_main,content_zeros);
  udl_batch_normalization_layer(conv5_block2_2_bn_desc, &t_conv5_block2_2_relu, &t_conv5_block2_2_conv, &t_conv5_block2_2_bn_b); 

  tensor4d_t t_conv5_block2_3_conv_k       = udl_tensor_from_buffer(2048,1,1,512,type_main, conv5_block2_3_conv_k);
  tensor4d_t t_conv5_block2_3_conv_b       = udl_tensor_from_buffer(1,1,2,2048,type_aux, conv5_block2_3_conv_b);
  conv2d_desc_t conv5_block2_3_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 1, .sy = 1};
  tensor4d_t t_conv5_block2_3_conv         = udl_tensor_create(1,7,7,2048,type_main,content_zeros);
  udl_conv2d_layer(conv5_block2_3_conv_desc, &t_conv5_block2_3_conv, &t_conv5_block2_2_relu, &t_conv5_block2_3_conv_k, &t_conv5_block2_3_conv_b);

  tensor4d_t t_conv5_block2_3_bn_b       = udl_tensor_from_buffer(1,1,2,2048,type_aux, conv5_block2_3_bn_b);
  batch_normalization_desc_t conv5_block2_3_bn_desc = {.active = active_type_linear};
  tensor4d_t t_conv5_block2_3_bn         = udl_tensor_create(1,7,7,2048,type_main,content_zeros);
  udl_batch_normalization_layer(conv5_block2_3_bn_desc, &t_conv5_block2_3_bn, &t_conv5_block2_3_conv, &t_conv5_block2_3_bn_b); 

  tensor4d_t t_conv5_block2_out         = udl_tensor_create(1,7,7,2048,type_main,content_zeros);
  add_desc_t conv5_block2_add_desc      = {.active = active_type_relu};
  udl_add_layer(conv5_block2_add_desc, &t_conv5_block2_out, &t_conv5_block1_out, &t_conv5_block2_3_bn);

  //block5-3
  tensor4d_t t_conv5_block3_1_conv_k       = udl_tensor_from_buffer(512,1,1,2048,type_main, conv5_block3_1_conv_k);
  tensor4d_t t_conv5_block3_1_conv_b       = udl_tensor_from_buffer(1,1,2,512,type_aux, conv5_block3_1_conv_b);
  conv2d_desc_t conv5_block3_1_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 1, .sy = 1};
  tensor4d_t t_conv5_block3_1_conv         = udl_tensor_create(1,7,7,512,type_main,content_zeros);
  udl_conv2d_layer(conv5_block3_1_conv_desc, &t_conv5_block3_1_conv, &t_conv5_block2_out, &t_conv5_block3_1_conv_k, &t_conv5_block3_1_conv_b);
  
  tensor4d_t t_conv5_block3_1_bn_b         = udl_tensor_from_buffer(1,1,2,512,type_aux, conv5_block3_1_bn_b);
  batch_normalization_desc_t conv5_block3_1_bn_desc = {.active = active_type_relu};
  tensor4d_t t_conv5_block3_1_relu         = udl_tensor_create(1,7,7,512,type_main,content_zeros);
  udl_batch_normalization_layer(conv5_block3_1_bn_desc, &t_conv5_block3_1_relu, &t_conv5_block3_1_conv, &t_conv5_block3_1_bn_b); 

  tensor4d_t t_conv5_block3_2_conv_k       = udl_tensor_from_buffer(512,3,3,512,type_main, conv5_block3_2_conv_k);
  tensor4d_t t_conv5_block3_2_conv_b       = udl_tensor_from_buffer(1,1,2,512,type_aux, conv5_block3_2_conv_b);
  conv2d_desc_t conv5_block3_2_conv_desc = {.active = active_type_linear, .px = 1, .py = 1, .sx = 1, .sy = 1};
  tensor4d_t t_conv5_block3_2_conv         = udl_tensor_create(1,7,7,512,type_main,content_zeros);
  udl_conv2d_layer(conv5_block3_2_conv_desc, &t_conv5_block3_2_conv, &t_conv5_block3_1_relu, &t_conv5_block3_2_conv_k, &t_conv5_block3_2_conv_b);

  tensor4d_t t_conv5_block3_2_bn_b         = udl_tensor_from_buffer(1,1,2,512,type_aux, conv5_block3_2_bn_b);
  batch_normalization_desc_t conv5_block3_2_bn_desc = {.active = active_type_relu};
  tensor4d_t t_conv5_block3_2_relu         = udl_tensor_create(1,7,7,512,type_main,content_zeros);
  udl_batch_normalization_layer(conv5_block3_2_bn_desc, &t_conv5_block3_2_relu, &t_conv5_block3_2_conv, &t_conv5_block3_2_bn_b); 

  tensor4d_t t_conv5_block3_3_conv_k       = udl_tensor_from_buffer(2048,1,1,512,type_main, conv5_block3_3_conv_k);
  tensor4d_t t_conv5_block3_3_conv_b       = udl_tensor_from_buffer(1,1,2,2048,type_aux, conv5_block3_3_conv_b);
  conv2d_desc_t conv5_block3_3_conv_desc = {.active = active_type_linear, .px = 0, .py = 0, .sx = 1, .sy = 1};
  tensor4d_t t_conv5_block3_3_conv         = udl_tensor_create(1,7,7,2048,type_main,content_zeros);
  udl_conv2d_layer(conv5_block3_3_conv_desc, &t_conv5_block3_3_conv, &t_conv5_block3_2_relu, &t_conv5_block3_3_conv_k, &t_conv5_block3_3_conv_b);

  tensor4d_t t_conv5_block3_3_bn_b       = udl_tensor_from_buffer(1,1,2,2048,type_aux, conv5_block3_3_bn_b);
  batch_normalization_desc_t conv5_block3_3_bn_desc = {.active = active_type_linear};
  tensor4d_t t_conv5_block3_3_bn         = udl_tensor_create(1,7,7,2048,type_main,content_zeros);
  udl_batch_normalization_layer(conv5_block3_3_bn_desc, &t_conv5_block3_3_bn, &t_conv5_block3_3_conv, &t_conv5_block3_3_bn_b); 

  tensor4d_t t_conv5_block3_out         = udl_tensor_create(1,7,7,2048,type_main,content_zeros);
  add_desc_t conv5_block3_add_desc      = {.active = active_type_relu};
  udl_add_layer(conv5_block3_add_desc, &t_conv5_block3_out, &t_conv5_block2_out, &t_conv5_block3_3_bn);

  //pool
  tensor4d_t t_avg_pool         = udl_tensor_create(1,1,1,2048,type_main,content_zeros);
  avgpooling2d_desc_t avg_pool_desc      = {.active = active_type_linear, .px = 0, .py = 0, .sx = 7, .sy = 7, .kw = 7, .kh = 7};
  udl_avgpooling2d_layer(avg_pool_desc, &t_avg_pool, &t_conv5_block3_out,  NULL);

  tensor4d_t t_predictions_k       = udl_tensor_from_buffer(1000,1,1,2048,type_main, predictions_k);
  tensor4d_t t_predictions_b       = udl_tensor_from_buffer(1,1,2,1000,type_aux, predictions_b);
  dense_desc_t predictions_desc = {.active = active_type_linear};
  tensor4d_t t_predictions         = udl_tensor_create(1,1,1,1000,type_main,content_zeros);
  udl_dense_layer(predictions_desc, &t_predictions, &t_avg_pool, &t_predictions_k, &t_predictions_b);

  tensor4d_t t_predictions_softmax  = udl_tensor_create(1,1,1,1000,type_main,content_zeros);
  udl_softmax_layer(&t_predictions_softmax, &t_predictions);
  
  s_t index = udl_tensor_argmax(&t_predictions_softmax);
  udl_printf("%s %f\n", imagenet_class_index[index], t_predictions_softmax.m[index]);
  return 0;
}