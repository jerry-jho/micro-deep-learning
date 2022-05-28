#include <udl.h>

#if UDL_SYMBOL_TYPE == UDL_SYMBOL_TYPE_32

m_t one_m() {
  return 1.0;
}

a_t one_a() {
  return 1.0;
}

m_t int_m(int x) {
  return (m_t)x;
}

m_t mul_mmm(m_t a, m_t b) {
  return a * b;
}

a_t mul_maa(m_t a, a_t b) {
  return a * b;
}

m_t mul_mam(m_t a, a_t b) {
  return a * b;
}

m_t div_mim(m_t a, s_t b) {
  if (b != 0) return a / b;
  return 0;
}

m_t div_mmm(m_t a, m_t b) {
  if (b != 0) return a / b;
  return 0;
}

m_t mac_mam(m_t a, a_t s, a_t b) {
  return a*s+b;
}

m_t add_mmm(m_t a, m_t b) {
  return a + b;
}

m_t active_function_relu(m_t a) {
  return a > 0 ? a : 0;
}

m_t sub_mmm(m_t a, m_t b) {
  return a - b;
}

m_t abs_mmm(m_t a) {
  return a > 0 ? a : (-1.0 * a);
}

m_t max_mmm(m_t a, m_t b) {
  return a > b ? a : b;
}

m_t min_mmm(m_t a, m_t b) {
  return a > b ? b : a;
}

m_t exp_mmm(m_t a) {
  unsigned int i=(1<<23)*(1.4426950409f*((double)a)+126.94201519f);
  return *((float *)(&i));
}

#endif

m_t active_function_linear(m_t a) {
  return a;
}

fmm_t get_active_function(active_t active) {
  fmm_t f;
  if (active == active_type_linear) {
    f = active_function_linear;
  } else if (active == active_type_relu) {
    f = active_function_relu;
  } else {
    return NULL;
  }
  return f;
}