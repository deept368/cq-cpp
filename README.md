# CQ-CPP
A C++ version of [Contextual Quantizer](https://github.com/yingrui-yang/ContextualQuantizer).

Implementation of Paper [Compact Token Representations with Contextual Quantization for Efficient Document Re-ranking](https://arxiv.org/pdf/2203.15328v1.pdf).


## Models
All the pretrained models are kept in **/model** folder. We are primarily using **.pt** files and loading them as models in the code. For **BERT**, we are using **model.proto** as pretrained weights. Along with this *vocabulary* and *codebook* files are also present in the **/model** folder.

## Installation instructions.

Refer the installation.pdf under the `Documentation` folder in this repository.

## Build
```bash
mkdir build 
cd build
cmake .. -DCMAKE_MODULE_PATH=/path/to/cq-cpp
make -j4
./bert-sample (This will be the entry point to run a sample code)
```

## Testing
The dataset used is MS MARCO passage dataset conatining 8.8 million passages. In **src/config.h**, we have two configs, **QUERY_FILE** and **RESULTS_FILE** where we put the path of the file that has all the queries of MS MARCO corresponding to their ids and path of the file which contains results obtained from topK retireval respectively.

## Results
A trec file is generated which contains the results of re-ranking. _Sample trec file can be found in /output folder_. 

## Presentation slides
https://docs.google.com/presentation/d/1gP35vJpDEbWEX8EkGIjHvcKbCkhJroiBGR8SWFYUQIs/edit#slide=id.g218157fd287_0_445


## Quick Info
The **main()** function resides in *bert-sample.cpp* file. All the necessary documentation fo various functions is included in the code.

## Thanks
**BERTCPP** model is taken from here [BERTCPP](https://github.com/LeeJuly30/BERTCpp) 

