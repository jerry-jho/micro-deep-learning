all:
	make -f ../../Makefile test
	./test

riscv:
	make -f ../../Makefile test TARGETS="gloss generic"\
		CROSS="riscv64-unknown-elf-" \
		CROSS_CFLAGS="-O2 -mcmodel=medany" \
		CROSS_LDFLAGS="-specs=htif.specs" 
	spike ./test
	
clean:
	make -f ../../Makefile clean
	rm -rf ./test