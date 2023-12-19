#include <torch/torch.h>
#include "../config.h"
#include "../utils.h"
#include <map>
#include <vector>

namespace lh
{
    /*
    Fetches and decodes codes for all the documents and generate approx embeddings for these documents.
    It used codebook and static embeddings to achieve the same.
    */
    class Decoder
    {
    public:
        explicit Decoder();
        ~Decoder();
        map<int, map<std::string, torch::Tensor> *> *decode(unordered_map<int, unordered_map<string, vector<pair<uint16_t, vector<uint8_t> *>> *> *> *fetched_codes);

    private:
        std::int64_t vocab_size_;
        std::int64_t dimension_size_;
        std::int64_t pad_token_id_;
        std::int64_t M_;
        std::int64_t K_;
        std::int64_t codebook_dim_;
        std::int64_t doc_maxlen_;

        vector<float> codebook_vector;

        torch::nn::EmbeddingImpl *non_contextual_embedding;
        torch::Tensor *codebook;
        torch::nn::LinearImpl *composition_layer;
    };
}
