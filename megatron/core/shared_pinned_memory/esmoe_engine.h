#ifndef _ESMOE_ENGINE_
#define _ESMOE_ENGINE_

#include "threadpool.h"
#include <mutex>
#include <vector>
#include <ATen/cuda/CUDAEvent.h>
#include <torch/extension.h>

class EsMoeEngine {
private:
    static EsMoeEngine* instance;
    bool configured_;
    
    EsMoeEngine () {}

public:

    ThreadPool thread_pool_;
    // std::vector<std::future<void>> lst_futures_;

    // std::unordered_map<std::string, std::future<std::variant<>>> future_dict;
    
    std::future<std::vector<torch::Tensor>> a2a_future;
    std::future<std::pair<std::vector<int>, std::vector<int>>> expert_future;
    std::future<torch::Tensor> input_future;

    EsMoeEngine(EsMoeEngine const&) = delete;
    void operator=(EsMoeEngine const&) = delete;
    ~EsMoeEngine () {}
    
    static EsMoeEngine& getInstance() {
        static EsMoeEngine instance;
        return instance;
    }
};

#endif