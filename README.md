# CQ-CPP
A C++ version of [Contextual Quantizer](https://github.com/yingrui-yang/ContextualQuantizer).

Implementation of Paper [Compact Token Representations with Contextual Quantization for Efficient Document Re-ranking](https://arxiv.org/pdf/2203.15328v1.pdf).


## Models
All the pretrained models are kept in **/model** folder. We are primarily using **.pt** files and loading them as models in the code. For **BERT**, we are using **model.proto** as pretrained weights. Along with this *vocabulary* and *codebook* files are also present in the **/model** folder.

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

## Testing
The dataset used is MS MARCO passage dataset conatining 8.8 million passages. In **src/config.h**, we have two configs, **QUERY_FILE** and **RESULTS_FILE** where we put the path of the file that has all the queries of MS MARCO corresponding to their ids and path of the file which contains results obtained from topK retireval respectively. _Sample data is present in /data folder of the remote box_

## Results
A trec file is generated which contains the results of re-ranking. _Sample trec file can be found in /output folder_. 

## Result


## Quick Info
The **main()** function resides in *bert-sample.cpp* file. All the necessary documentation fo various functions is included in the code.

## Thanks
**BERTCPP** model is taken from here [BERTCPP](https://github.com/LeeJuly30/BERTCpp) 
