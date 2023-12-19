import sys
import ujson
import torch
import numpy as np
import pandas as pd
import hashlib
import argparse
from tqdm import tqdm
import gc
import bitarray
import os

# path of the directory for data
# path = "CQ_sample\index_sample\index_colbertv2_prduct_static_1layer\\"
path = "./test230816_16_16/"
number_of_docs = 256

# argparse settings
# parser = argparse.ArgumentParser(description='input file')
# parser.add_argument('--file1', type=str, default='doclens.0.json')
# parser.add_argument('--file2', type=str, default='0.pt')

# args = parser.parse_args()
# # print(args)
# file1 = args.file1
# file2 = args.file2

indexs = [_ for _ in range(94)]

for index in indexs:

# !!!!!!!!! indent from here !!!!!!!!!!

    file1 = 'doclens.'+str(index)+'.json'
    file2 = str(index)+'.pt'
    print('======================'+file2+'======================\n')

    # read a document file and load documents
    # in the future you need to use key-value to fetch
    # for each token, the 0th element is tokenid, then 1-m elements are the codes.
    doclen=ujson.load(open(path+file1))#[0] # doclens.0.json
    # doclen = doclen[:12] # alter the number of testcase
    keys = [i for i in range(len(doclen))] # generate the key for each document


    # to store the keys and encrypted keys
    key_map = {}

    # seperate the documents into 10 groups using MOD
    for remainder in tqdm(range(number_of_docs)):
    # testcase
    # for remainder in [3]:
        # to decide which section of file should be processed
        left = 0
        right = 0

        # to store the keys, length, offsets and values
        keys_ = []
        keys_dec_ = []
        len_ = []
        values_ = []

        total_length = 0

        p = torch.load(path+file2, map_location='cpu')

        for length,key in tqdm(zip(doclen,keys)):
            if key % number_of_docs != remainder:
                right += length
                left += length
            else: 
                # find the tokens of a certain key
                right += length
                # p0 = torch.load(path+file2, map_location='cpu')[left:right,:]
                p0 = p[left:right,:]
                left += length

                total_length += length

                # iteratively calculate the value of tokens and concatenate them
                p0 = np.array(p0)
    #             print(p0) # all the numbers in p0 are integer
                value = ''
                for x in p0:
                    cnt = 0
                    for y in x: #token: len = 144 = 16 + 16x8
                        cnt += 1
                        if cnt==1:
                            tmp = '{:016b}'.format(y) # token id, 2 bytes
                        else:
                            tmp = '{:08b}'.format(y) # code in codebooks, 1 byte
                        value += str(tmp)
                
                # encrypt the key, store the encrypted key and their mapping relation
                md = hashlib.md5(str(key).encode())
                key_hex = md.hexdigest() # hex, 32 characters, 128 bytes
    #             print(key_hex) 
                key_bin = '{:032b}'.format(int(key_hex[:8], 16)) # convert the first 8 character into bin, 32 bits, 4 bytes
    #             print(key_bin)
                keys_.append(key_bin)
                keys_dec_.append(key)
                
                len_.append(length*144) # the length of value
                values_.append(value)
        
        # print('number of tokens:', total_length, '----------')

    #     print(keys_)
    #     print(len_)
    #     print(values_)

        # store the map between key and key_bin
        df = pd.DataFrame()
        df['keys_'] = keys_ # bin
        df['keys_dec'] = keys_dec_ # dec
        df.to_csv(os.path.join('keys_16_16', 'key_map_{}'.format(remainder)), index=None)
        # del df
        # gc.collect()

        # calculate the offset for each value
        offsets_ = [0 for _ in range(len(keys_))]
        offset = 128*8 + 4*8*2*len(keys_) # metadata 128 bytes + (key 4 bytes + offset 4 bytes) * number of keys
        for i in range(len(len_)):
    #         print(offset)
            offsets_[i] = '{:032b}'.format(offset) #offset, 4 bytes
            offset += len_[i]
    #     print(offsets_)
        # del len_
        # gc.collect()
    
        # generate the metadata
        # Assuming that metadata occupies 128 bytes
        metadata = ''
        M = '{:032b}'.format(16) # M = 16, 4 bytes
        K = '{:032b}'.format(8) # K = 8, 4 bytes
        N_key = '{:032b}'.format(len(keys_)) # number of documents, 4 bytes
        B_key = '{:032b}'.format(4) # number of bytes per document key, 4 bytes
        B_offset = '{:032b}'.format(4) # number of bytes per offset, 4 bytes
        B_token = '{:032b}'.format(18) # number of bytes per token, 4 bytes

        metadata = M + K + N_key + B_key + B_offset + B_token + 104*8*'0'
    #     print(metadata) 
        
        # calculate the final result (the concatenation of metadata, keys, offsets and values)
        result = ''
        result += metadata #calculate before
        for key, offset in zip(keys_, offsets_):
            result += key
            result += offset
        print('\nsize of the prefix:', len(result))
        for value in values_:
            result += value
        print('\nsize of the final index:', len(result))

        bits = bitarray.bitarray(result)
        with open(os.path.join('output_result_16_16', 'result_{}'.format(remainder)),'wb') as f:
            bits.tofile(f)

            
        

        # result = bytes(result, encoding = "utf8") # convert from string to bytes
        # result = bytearray(result, 'utf-8')

        # store the results into ten different files
        # fh = open('result_{}_{}'.format(file2, remainder), 'wb')
        # fh = open('result_{}_{}'.format(file2, remainder), 'w')
        # fh.write(result)
        # fh.close()