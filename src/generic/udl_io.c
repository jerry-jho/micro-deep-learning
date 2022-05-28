#include <udl.h>

result_t udl_load(void ** dest, l_t trunk, l_t count, void * source) {
  * dest = source;
  return RESULT_OK;
}

void udl_putc(char c) {
}

int udl_puts(char * s) {
  return 0;
}