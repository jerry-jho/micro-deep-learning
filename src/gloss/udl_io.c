#include <udl.h>
#include <stdio.h>
#include <string.h>

result_t udl_load(void ** dest, l_t trunk, l_t count, void * source) {
  *dest = source;
  return RESULT_OK;
}

void udl_putc(char c) {
  putc(c, stdout);
}

int udl_puts(char * s) {
  printf("%s", s);
  return strlen(s);
}