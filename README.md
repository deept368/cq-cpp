# CQ-CPP
A C++ version of [Contextual Quantizer](https://github.com/yingrui-yang/ContextualQuantizer).

Implementation of Paper [Compact Token Representations with Contextual Quantization for Efficient Document Re-ranking](https://arxiv.org/pdf/2203.15328v1.pdf).


## Models
All the pretrained models are kept in **/model** folder. We are primarily using **.txt** files and loading them as models in the code. For **BERT**, we are using **model.proto** as pretrained weights. Along with this *vocabulary* and *codebook* files are also present in the **/model** folder. **model.proto** file is big and can't be pushed to github. I caan be downloaded from here [model]()

## Dependencies

### Protobuf
Bert uses **protobuf** to convert pytorch pretrained model in protobuf *(.proto)* file and load it in C++.
Make sure to download protobuf. 
One of the ways to install protobuf: **pip install protobuf**

### MKL
Bert uses **MKL** to implement bias operator.
To install: **pip install mkl**
Make sure that the */bin* folder of MKL should be present in **/opt/intel/mkl**, otherwise you might need to make changes in *CMakeLists.txt*


### utf8proc
Bert uses **utf8proc** to process input string.
To install: **sudo apt-get install libutf8proc-dev**

### libtorch
We are using PyTorch C++ to carry out various neural netowrk operations. 
Install the stable version of **libtorch** for C++ from here: [C++ Pytorch](https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcpu.zip). *libtorch* should be present in **/cq-cpp**.


## Build
```bash
mkdir build 
cd build
cmake .. -DCMAKE_MODULE_PATH=/path/to/cq-cpp
make -j4
./bert-sample (This will be the entry point to run a sample code)
```
## Quick Info
The **main()** function resides in *bert-sample.cpp* file. All the necessary documentation fo various functions is included in the code.

## Thanks
**BERTCPP** model is taken from here [BERTCPP](https://github.com/LeeJuly30/BERTCpp) 
