#include <udl.h>

result_t udl_load(void ** dest, uint64_t trunk, uint64_t count, void * source) {
  * dest = source;
  return RESULT_OK;
}

void udl_putc(char c) {
  *((volatile uint32_t *)(0x8)) = c; // putc
  *((volatile uint32_t *)(0x4)) = 0x4; // putc
}

int udl_puts(char * s) {
  *((volatile uint32_t *)(0x8)) = (uint32_t)(s); // puts
  *((volatile uint32_t *)(0x4)) = 0x8; // puts
  return 0;
}