MAIN_MAKEFILE:=../../Makefile

all:
	make -f $(MAIN_MAKEFILE) test
	./test

avx:
	make -f $(MAIN_MAKEFILE) test TARGETS="avx os generic" CROSS_CFLAGS="-mavx2"
	./test

riscv:
	make -f $(MAIN_MAKEFILE) test TARGETS="gloss generic"\
		CROSS="riscv64-unknown-elf-" \
		CROSS_CFLAGS="-O2 -mcmodel=medany" \
		CROSS_LDFLAGS="-specs=htif.specs" 
	spike ./test
	
clean:
	make -f $(MAIN_MAKEFILE) clean
	rm -rf ./test