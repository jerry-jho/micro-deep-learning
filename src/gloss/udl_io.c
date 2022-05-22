#include <udl.h>
#include <stdio.h>
#include <string.h>

result_t udl_load(void ** dest, uint64_t trunk, uint64_t count, void * source) {
  *dest = source;
  return RESULT_OK;
}

void udl_putc(char c) {
  printf("%c", c);
}

int udl_puts(char * s) {
  printf("%s", s);
  return strlen(s);
}