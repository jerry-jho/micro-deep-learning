#include <udl.h>

int main(int argc, char ** argv) {
  udl_printf("Hello World! %d-int %x-hex %f-float %f-float %f-float\n", 123, 12, 45.112, 0.000123, 0.0);
  return 0;
}