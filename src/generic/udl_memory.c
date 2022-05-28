#include <udl.h>

void * udl_malloc(l_t size) {
  return NULL;
}

void udl_free(void * v) {
}

void udl_memcpy(void * dst, void * src, l_t size) {
  char * d = (char *)dst;
  char * s = (char *)src;
  while(size--) {
    *(d++) = *(s++);
  }
}