#include <udl.h>

void * udl_malloc(uint64_t size) {
  return NULL;
}

void udl_free(void * v) {
}

void udl_memcpy(void * dst, void * src, uint64_t size) {
  char * d = (char *)dst;
  char * s = (char *)src;
  while(size--) {
    *(d++) = *(s++);
  }
}