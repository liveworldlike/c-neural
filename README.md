# c-neural
A small neural network written in C. Performs digit recognition on the MNIST dataset.

## How to Run
1. Create `dataset` directory.
2. Download [dataset files](http://yann.lecun.com/exdb/mnist/), and place decompressed files into `dataset/`.
3. Compile:
```bash
gcc main.cpp matrix.cpp mnist.cpp net.cpp -O2 -Wall -o run
```
4. Run
```bash
./run
```
