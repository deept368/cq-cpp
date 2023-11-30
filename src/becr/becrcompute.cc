#include "becrcompute.h"

namespace lh{
   
    BecrCompute::BecrCompute(){
        vocab_size_ = VOCAB_SIZE;
        query_maxlen = QUERY_MAXLEN;
        dimension_size_ = DIMENSION_SIZE;
        pad_token_id_ = PAD_TOKEN_ID; 

        // bert_ = new Bert<T>(names, graph, pre_batch_size, pre_seq_len, hidden_size_, num_heads, head_hidden_size, intermediate_ratio, num_layers);
        tokenizer_ = new FullTokenizer("../model/vocab.txt");

        // //static embedding is fetched for all the vocabulary from a .pt file into a tensor [30522(vocab_size) * 128(embedding_dim)]
        // torch::Tensor* static_embeddings = new torch::Tensor();
        // torch::load(*static_embeddings, "../model/non_contextual_embeddings.pt");

        //  //torch::nn::EmbeddingImpl(PyTorch C++) model is initialised and static_embeddings tensor is loaded as pretrained weight. 
        // auto embedding_options = torch::nn::EmbeddingOptions(vocab_size_, dimension_size_).padding_idx(pad_token_id_)._weight(*static_embeddings);
        // non_contextual_embedding = new torch::nn::EmbeddingImpl(embedding_options);

        // cout << "Successfully loaded unigrams" << endl;


        // // Load data from a JSON file
        // std::ifstream bigram_file("path/to/bigram_mapping_emb_dict.json");
        // std::string bigram_str((std::istreambuf_iterator<char>(bigram_file)), std::istreambuf_iterator<char>());
        // std::unordered_map<std::string, std::unordered_map<std::string, std::vector<float>>> bigram_emb = torch::data::serialization::fromJSON<std::unordered_map<std::string, std::unordered_map<std::string, std::vector<float>>>>(bigram_str);


        unigram_emb = new torch::Tensor();
        torch::load(*unigram_emb, "../model/non_contextual_embeddings.pt");
        // std::ifstream bigram_file("../../bigram_mapping_emb_dict.json");
        // bigram_file >> bigram_emb;
        cout << "Successfully loaded unigram embeddings." << endl;

        // Load data from a JSON file
        std::ifstream file("../data/bigram_mapping_emb_dict.json");
        if (!file.is_open()) {
            std::cerr << "Error opening the JSON file" << std::endl;
        }

        try {
            // Parse JSON from the file
            json j;
            file >> j;

            // Extract data and populate your data structure
            bigram_emb = j.get<decltype(bigram_emb)>();
        } catch (const std::exception& e) {
            std::cerr << "Error parsing JSON: " << e.what() << std::endl;
        }
    
        std::cout << "Size of bigram_emb: " << bigram_emb.size() << std::endl;
        // auto sample = bigram_emb["9440"]["1998"];
        // for (auto& l: sample)
        // {
        //     cout << "New line: " << endl;
        //     for (auto val: l)
        //         cout << val << " ";
        //     cout << endl;
        // }

        // // Print keys of bigram_emb (first 10 keys)
        // int count = 0;
        // std::cout << "Keys of bigram_emb: ";
        // for (const auto& key : bigram_emb) {
        //     if (count < 10) {
        //         std::cout << key.first << " ";
        //         count++;
        //     } else {
        //         break;
        //     }
        // }
        // std::cout << std::endl;

        // Access the first element in bigram_emb
        // auto first = bigram_emb.begin()->second;
        // std::cout << "Size of first: " << first.size() << std::endl;

        // // Access the second element in first
        // auto second = first.begin()->second;
        // std::cout << "Size of second: " << second.size() << std::endl;

        // // Print the first 10 keys of second
        // count = 0;
        // std::cout << "Keys of second: ";
        // for (const auto& key : first) {
        //     if (count < 10) {
        //         std::cout << key.first << " ";
        //         count++;
        //     } else {
        //         break;
        //     }
        // }
        // std::cout << std::endl;

        // // Print the sizes of the first two vectors in second
        // if (!second.empty()) {
        //     std::cout << "Size of the first vector in second: " << second[0].size() << std::endl;
        //     std::cout << "Size of the second vector in second: " << second[1].size() << std::endl;
        // }

        // // Print the values of second
        // std::cout << "Values of second: " << second[0][0] << ", " << second[1][0] << std::endl;


        cout << "Successfully loaded bigram embeddings." << endl;

    }

    BecrCompute::~BecrCompute(){
        delete non_contextual_embedding;
        delete unigram_emb;
        delete tokenizer_;
        
    }    

    /**
     Computes the BERT embedding of given input strings. Treats query and document differently. Currently support is ony added 
     for queries as this is only required for CQ.
     @param input_string the input strings to encode
     @return linear a vector containing the BERT embeddings of the input strings of size (BATCH_SIZE * QUERY_MAXLEN * HIDDEN_DIM_SIZE(768)) 
    */
    torch::Tensor BecrCompute::compute(std::vector<std::string>* input_string){
        
        //computing the batch size
        int curr_batch_size = input_string->size();

        //necessary tokens are added and the input strings are converted to tokens
        std::vector<std::vector<std::string>>* input_tokens = new std::vector<std::vector<std::string>>(curr_batch_size);
        for (std::size_t i = 0; i < curr_batch_size; i++){
            tokenizer_->tokenize((*input_string)[i].c_str(), &(*input_tokens)[i], query_maxlen);
            (*input_tokens)[i].insert((*input_tokens)[i].begin(), "[CLS]");    
            (*input_tokens)[i].push_back("[SEP]");
        }

        //mask is computed and padding is applied to input strings
        uint64_t* mask = new uint64_t[curr_batch_size];
        for (std::size_t i = 0; i < curr_batch_size; i++){
            mask[i] = (*input_tokens)[i].size();
            for (int j = (*input_tokens)[i].size(); j < query_maxlen; j++){
                (*input_tokens)[i].push_back("[MASK]");
            }
        }

        //token ids are computed using tokens. vocab.txt is used for the same.
        uint64_t* input_ids = new uint64_t[curr_batch_size * query_maxlen];
        for (std::size_t i = 0; i < curr_batch_size; i++)
            tokenizer_->convert_tokens_to_ids((*input_tokens)[i], input_ids + i * query_maxlen);
        
        std::vector<string>* tokens = new std::vector<string>(curr_batch_size * query_maxlen);
        for (std::size_t i = 0; i < curr_batch_size * query_maxlen; ++i) {
            (*tokens)[i] = to_string(input_ids[i]);
        }


        // Initialize representations and weights
        torch::Tensor reps = torch::Tensor(torch::zeros({static_cast<long>(tokens->size()), dimension_size_}));
        torch::Tensor weights = torch::ones({static_cast<long>(tokens->size())}) / (1 + static_cast<long>(tokens->size()));

        // Update representations and weights using bigram embeddings
        for (int i = 0; i < static_cast<int>(tokens->size()); ++i)
        {
            for (int j = i + 1; j < std::min(static_cast<int>(tokens->size()), i + WINDOW_SIZE); ++j)
            {
                auto it1 = bigram_emb.find((*tokens)[i]);
                if (it1 != bigram_emb.end())
                {
                    auto it2 = it1->second.find((*tokens)[j]);
                    if (it2 != it1->second.end())
                    {
                        auto temp = torch::tensor(it2->second[0]) / (j - i);
                        reps[i] += torch::tensor(it2->second[0]) / (j - i);
                        reps[j] += torch::tensor(it2->second[1]) / (j - i);
                        weights[i] += 1 / (j - i);
                        weights[j] += 1 / (j - i);
                    }
                }
            }
        }

        // Normalize weights
        weights = 1 / weights;

        // Compute the final representation
        // reps = weights.unsqueeze(1) * reps;

        reps = reps.view({-1, dimension_size_});

        // Compute the final representation
        reps = weights.view({-1, 1}) * reps;

        // Reshape the final tensor to have the correct batch size
        reps = reps.view({curr_batch_size, -1, dimension_size_});
  

        delete[] input_ids;
        delete input_tokens;
        delete tokens;

        // reps = reps.unsqueeze(0);
        return reps;
    }
}
