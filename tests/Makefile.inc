DATA_FILE?=
MAIN_MAKEFILE:=../../Makefile

all:
	make -f $(MAIN_MAKEFILE) test DATA_FILE=$(DATA_FILE)
	time ./test

avx:
	make -f $(MAIN_MAKEFILE) test TARGETS="avx os generic" CROSS_CFLAGS="-mavx2" DATA_FILE=$(DATA_FILE)
	time ./test

riscv:
	make -f $(MAIN_MAKEFILE) test TARGETS="gloss generic"\
		CROSS="riscv64-unknown-elf-" \
		CROSS_CFLAGS="-O2 -mcmodel=medany" \
		CROSS_LDFLAGS="-specs=htif.specs" \
		DATA_FILE=$(DATA_FILE)
	time spike ./test
	
clean:
	make -f $(MAIN_MAKEFILE) clean
	rm -rf ./test