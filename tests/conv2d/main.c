#include <udl.h>

int main(int argc, char ** argv) {
  tensor4d_t ifm = udl_tensor_create(1,2,2,4,type_main, content_linspace);
  tensor4d_t ker = udl_tensor_create(2,1,1,4,type_main, content_linspace);
  tensor4d_t ofm = udl_tensor_create(1,2,2,2,type_main, content_empty);
  conv2d_desc_t desc = {.px = 0, .py = 0, .sx = 1,.sy = 1,.active = active_type_linear};
  udl_tensor_print(&ifm);
  udl_tensor_print(&ker);
  udl_conv2d_layer(desc, &ofm, &ifm, &ker, NULL);
  udl_tensor_print(&ofm);
  return 0;
}