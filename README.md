# Micro Deep Learning Framework, Pure C99

## Native Requirements

Windows 10/11 WSL, MSYS2 or Any Linux with

1. GNU GCC (Add to $PATH)
2. sh, make, rm, sed

Example on Ubuntu 20.04

    apt update
    apt install make gcc

## Native Tests

    cd tests/<testname>
    make

## Cross Compile on RISC-V (Ubuntu 20.04)

### Install packages
    
    apt install device-tree-compiler libboost-all-dev

### Install riscv toolchain

Please install pre-compiled riscv toolchain from SiFive (not from apt or the official binary release). Recommended version: riscv64-unknown-elf-gcc-10.1.0-2020.08.2-x86_64-linux-ubuntu14.

    $ tar zxf riscv64-unknown-elf-gcc-10.1.0-2020.08.2-x86_64-linux-ubuntu14.tgz
    $ ln -s $PWD/riscv64-unknown-elf-gcc-10.1.0-2020.08.2-x86_64-linux-ubuntu14 /opt/riscv
    $ export PATH=/opt/riscv/bin:$PATH

The reason is, spike/libgloss requires libc with -mcmodel=medany configuration (objects locate 0x80000000 and higher) and SiFive libc is configured this way.


### Install spike

    $ git clone https://github.com/riscv-software-src/riscv-isa-sim.git
    $ cd riscv-isa-sim
    $ mkdir build
    $ cd build
    $ ./configure --prefix=/opt/riscv
    $ make
    $ make install

### install libgloss-htif

    $ git clone https://github.com/ucb-bar/libgloss-htif.git
    $ cd libgloss-htif
    $ mkdir build
    $ cd build
    $ ../configure --prefix=/opt/riscv/riscv64-unknown-elf --host=riscv64-unknown-elf
    $ make
    $ make install

### test

    $ cd tests/<testname>
    $ make clean
    $ make riscv





