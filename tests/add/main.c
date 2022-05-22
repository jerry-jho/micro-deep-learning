#include <udl.h>

int main(int argc, char ** argv) {
  tensor4d_t t1 = udl_tensor_create(1,1,1,4,type_main, content_linspace);
  tensor4d_t t2 = udl_tensor_copy(&t1);
  tensor4d_t t3 = udl_tensor_create_alike(&t1,content_empty);
  udl_tensor_print(&t1);
  udl_tensor_print(&t2);
  udl_add_layer(active_type_linear, &t1, &t2, &t3);
  udl_tensor_print(&t3);
  return 0;
}