# resnet50 test

The resnet50 weight is too large to check in to the repo, so you have to export the weight yourself.

Just install python and tensorflow, then

    cd py
    python3 main.py
    python3 export_udl.py

and

    cd ..
    make


Please note this test cannot be run on riscv-spike by now.

# copyright

the py/elephant.jpg is taken by the author.