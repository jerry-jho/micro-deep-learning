#include <udl.h>

#include <stdarg.h>

static l_t __pow(int x,int y) {
  l_t sum = 1;
  while(y--) {
    sum *= x;
  }
  return sum;
}

static int __print_int(int dec) {
  int count = 0, ret = 0;
  int r_val = dec;
  char ch;
  if (dec == 0) {
    udl_putc('0');
    return 1;
  }
  while(r_val) {
    count++;
    r_val /= 10;
  }
  ret = count;
  r_val = dec;
  while(count) {
    ch = r_val / __pow(10, count - 1);
    r_val %= __pow(10, count - 1);
    udl_putc(ch + '0');
    count--;
  }
  return ret;
}

static int __print_int_hex(int dec) {
  int count = 0;
  int r_val = dec;
  char ch;
  if (dec == 0) {
    udl_putc('0');
    return 1;
  }
  if (dec < 0) {
    udl_putc('-');
    count ++;  
    dec = -1 * dec;  
  }
  r_val = dec;
  while(r_val) {
    count++;
    r_val /= 16;
  }
  r_val = dec;
  while(count) {
    ch = r_val / __pow(16, count - 1);
    r_val %= __pow(16, count - 1);
    if(ch <= 9) udl_putc(ch + '0');
    else udl_putc(ch - 10 + 'a');
    count--;
  }
  return count;
}

static int __print_float(float flt) {
  int scount = 0, countint = 0,countflt = 0;
  int count = 0, r_val = 0;
  int tmpint;
  int tmpflt;

  if (flt < 0) {
    udl_putc('-');
    flt = -1 * flt;
    scount = 1;  
  }
  tmpint = (int)flt;
  tmpflt = (long int)(100000000 * (flt - tmpint));
  if(tmpflt % 10 >= 5) {
    tmpflt = tmpflt / 10 + 1;
  } else {
    tmpflt = tmpflt / 10;
  }
  r_val = tmpflt;
  
  while(r_val) {
    count++;
    r_val /= 10;
  }

  countint = __print_int(tmpint);
  udl_putc('.');
 
  for(int i = 0; i < 7 - count; i++) {
    udl_putc('0');
  }
  if (tmpflt != 0) {
    countflt = __print_int(tmpflt);
  }
  return scount + countint + 1 + count + countflt;
}

int udl_printf(const char *str, ...) {
  va_list ap;
  int val;
  float val_float;
  char count;
  char *s = NULL;
  int res = 0;
  va_start(ap, str);
  while('\0' != *str) {
    switch(*str) {
      case '%':
        str++;
        switch(*str) {
          case 'd':
            val = va_arg(ap, int);
            count = __print_int(val);
            res += count;
            break;
          case 'x':
            val = va_arg(ap, int);
            count = __print_int_hex(val);
            res += count;
            break;
          case 'f':
            val_float = va_arg(ap, double);
            count = __print_float(val_float);
            res += count;
            break;
          case 's':
            s = va_arg(ap, char*);
            res += udl_puts(s);
            break;
          case 'c':
            udl_putc((char)va_arg(ap, int));
            res += 1;
            break;
          default:
            udl_putc('%');
            if (*str == '\0') {
              res += 1;
              goto end;
            }
            udl_putc(*str);
            res += 2;
            break;
        }
        break;
      default:
        udl_putc(*str);
        res += 1;
        break;
    }
    str++;
  }
end:
  va_end(ap);
  return res;
}
 