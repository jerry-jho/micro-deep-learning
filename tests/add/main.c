#include <udl.h>

int main(int argc, char ** argv) {
  float buf[4] = {4., 8., 10., 12.};
  tensor4d_t t1 = udl_tensor_create(1,1,1,4,type_main, content_linspace);
  tensor4d_t t2 = udl_tensor_from_buffer(1,1,1,4,type_main,buf);
  tensor4d_t t3 = udl_tensor_create_alike(&t1,content_empty);
  add_desc_t desc = {.active = active_type_linear};
  udl_tensor_print(&t1);
  udl_tensor_print(&t2);
  udl_add_layer(desc, &t3, &t1, &t2);
  udl_tensor_print(&t3);
  return 0;
}