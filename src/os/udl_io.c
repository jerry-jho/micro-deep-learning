#include <udl.h>
#include <stdio.h>

result_t udl_load(void ** dest, uint64_t trunk, uint64_t count, void * source) {
  FILE * p = fopen((const char *)source, "r");
  if (p) {
    if (*dest == NULL) {
      *dest = udl_malloc(trunk * count);
    }
    uint64_t n = fread(*dest, trunk, count, p);
    if (n == count) return RESULT_OK;
    else return RESULT_ERROR_SHAPE_MISMATCH;
  }
  return RESULT_ERROR_FILE_NOT_FOUND;
}

void udl_putc(char c) {
  putc(c, stdout);
}

int udl_puts(char * s) {
  int i = 0;
  while(*s) {
    udl_putc(*s++);
    i++;
  }
  return i;
}