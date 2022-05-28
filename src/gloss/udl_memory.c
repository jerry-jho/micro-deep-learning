#include <udl.h>
#include <stdlib.h>
#include <string.h>

void * udl_malloc(l_t size) {
  return malloc(size);
}

void udl_free(void * v) {
  free(v);
}

void udl_memcpy(void * dst, void * src, l_t size) {
  memcpy(dst, src, size);
}