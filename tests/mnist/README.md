# mnist test

The cdata directory contains pre-trained weight. Just type 

    make

or

    make riscv

to see the inference result.


If you would like to re-train the weight, please install python and pytorch, and run

    cd py
    python3 train.py

The weight will be saved as file cnn.pth, you may run

    python3 test.py

to check the trained weight and get the vector data of the last layer.

Then you run

    python3 export_udl.py

to convert the weight into c-array named data.c, and copy data.c to cdata.