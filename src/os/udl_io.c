#include <udl.h>
#include <stdio.h>
#include <string.h>

result_t udl_load(void ** dest, l_t trunk, l_t count, void * source) {
  FILE * p = fopen((const char *)source, "r");
  if (p) {
    if (*dest == NULL) {
      *dest = udl_malloc(trunk * count);
    }
    l_t n = fread(*dest, trunk, count, p);
    if (n == count) return RESULT_OK;
    else return RESULT_ERROR_SHAPE_MISMATCH;
  }
  return RESULT_ERROR_FILE_NOT_FOUND;
}

void udl_putc(char c) {
  putc(c, stdout);
}

int udl_puts(char * s) {
  printf("%s", s);
  return strlen(s);
}