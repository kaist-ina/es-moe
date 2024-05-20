#include <iostream>
#include <torch/extension.h>
#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>


// CUDA utilities and system includes
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/ATen.h>
#include <ATen/native/cuda/Resize.h>
#include <cuda_fp16.h>

#include <sys/mman.h>
#include <fcntl.h>
#include <sys/shm.h>
#include <future>
#include "esmoe_engine.h"

#define KEY_NUM 1000000

torch::Tensor shared_pinned_memory(torch::Tensor tensor, int rank, int layer, int expert, int order, int type, bool pinning = true){
  // int shm_id;
  auto key_val = KEY_NUM * type;
  auto tensor_size = tensor.numel() * tensor.element_size();

  std::string str = std::to_string(key_val + layer * 10000 + expert * 10 + order);
  const char* cstr = str.c_str();

  int fd = shm_open(cstr, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  if (fd == -1) {
      std::cerr << cstr <<", Failed to create/open shared memory object: " << std::strerror(errno) << std::endl;
      exit(0);
  }

  if (ftruncate(fd, tensor_size) == -1) {
      std::cout << "ftruncate failed\n" << std::endl;
      exit(0);
  }

  void *memory_segment = mmap(NULL, tensor_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (memory_segment == MAP_FAILED) {
      std::cout << "mmap failed\n" << std::endl;
      exit(0);
  }

  // NOTE: Only Params and Grads should be pinned
  if (pinning && (type == 1 || type == 2)){
      C10_CUDA_CHECK(cudaHostRegister((void *) memory_segment, (size_t) tensor_size, cudaHostRegisterMapped));
  }

  auto shared_tensor = torch::from_blob(memory_segment, tensor.sizes(), tensor.dtype());

  // NOTE: Only Params need initialization
  if ((rank == 0) && (type == 1)){
      shared_tensor.copy_(tensor);
  }

  return shared_tensor;
}

// Get as a list of params and upload them in the Thread function
void upload_params(torch::Tensor cpu_param, torch::Tensor gpu_param) {
  auto tensor_bytes = cpu_param.numel() * sizeof(torch::kFloat16) * 2; //sizeof(torch::kFloat32);
  at::native::resize_bytes_cuda(gpu_param.storage().unsafeGetStorageImpl(), tensor_bytes);
//   gpu_param.copy_(cpu_param.to(gpu_param.device(), true));

    //printf("[%d] %x %x\n", getpid(), cpu_param.data_ptr(), gpu_param.data_ptr());
    // std::cout << "[" ]" cpu_param.data_ptr() << std::endl;
  gpu_param.copy_(cpu_param, true);
}

void upload_tensor(torch::Tensor cpu_tensor, torch::Tensor gpu_tensor,
                            unsigned long long stream_ptr) {

    const auto current_device = c10::cuda::current_device();
    at::cuda::CUDAStream stream = at::cuda::getStreamFromExternal(reinterpret_cast<cudaStream_t>(stream_ptr), current_device);
    at::cuda::CUDAStreamGuard stream_guard(stream);
    auto tensor_bytes = cpu_tensor.numel() * cpu_tensor.element_size();
    at::native::resize_bytes_cuda(gpu_tensor.storage().unsafeGetStorageImpl(), tensor_bytes);
    gpu_tensor.copy_(cpu_tensor, true);
}

void offload_tensor(torch::Tensor gpu_tensor, torch::Tensor cpu_tensor,
                            unsigned long long stream_ptr) {

    const auto current_device = c10::cuda::current_device();
    at::cuda::CUDAStream stream = at::cuda::getStreamFromExternal(reinterpret_cast<cudaStream_t>(stream_ptr), current_device);
    at::cuda::CUDAStreamGuard stream_guard(stream);
    cpu_tensor.copy_(gpu_tensor, true);
    at::native::resize_bytes_cuda(gpu_tensor.storage().unsafeGetStorageImpl(), 0);
}

void upload_experts_params(std::vector<torch::Tensor> cpu_param_vec, 
                            std::vector<torch::Tensor> gpu_param_vec,
                            unsigned long long stream_ptr){
    nvtxRangePush("upload_experts_params");
    const auto current_device = c10::cuda::current_device();
    at::cuda::CUDAStream stream = at::cuda::getStreamFromExternal(reinterpret_cast<cudaStream_t>(stream_ptr), current_device);
    {
        at::cuda::CUDAStreamGuard stream_guard(stream);
        for (int i = 0; i < cpu_param_vec.size(); i++){
            upload_tensor(cpu_param_vec[i], gpu_param_vec[i], stream_ptr);
        }
    }
    nvtxRangePop();
}

void offload_grads(torch::Tensor cpu_grads, 
                    torch::Tensor gpu_grads, 
                    unsigned long long stream_ptr, 
                    float multiply_factor) {
  const auto current_device = c10::cuda::current_device();
  at::cuda::CUDAStream stream = at::cuda::getStreamFromExternal(reinterpret_cast<cudaStream_t>(stream_ptr), current_device);
  at::cuda::CUDAStreamGuard stream_guard(stream);
  auto gpu_grads_fp32 = gpu_grads.to(cpu_grads.dtype());
  //gpu_grads_fp32.mul_(multiply_factor);
  gpu_grads.record_stream(stream);

  cpu_grads.copy_(gpu_grads_fp32, true);
  gpu_grads_fp32.record_stream(stream);
}

void offload_experts_grads(std::vector<torch::Tensor> cpu_grads_vec, 
                            std::vector<torch::Tensor> gpu_grads_vec,
                            unsigned long long stream_ptr, 
                            float multiply_factor){

    const auto current_device = c10::cuda::current_device();
    at::cuda::CUDAStream stream = at::cuda::getStreamFromExternal(reinterpret_cast<cudaStream_t>(stream_ptr), current_device);
    {
        at::cuda::CUDAStreamGuard stream_guard(stream);
        for (int i = 0; i < cpu_grads_vec.size(); i++){
            offload_grads(cpu_grads_vec[i], gpu_grads_vec[i], stream_ptr, multiply_factor);
            //offload_tensor(gpu_grads_vec[i], cpu_grads_vec[i], stream_ptr);
        }
    }
}

void free_params(std::vector<torch::Tensor> gpu_params_vec, unsigned long long stream_ptr)
{
    nvtxRangePush("free_params");
    int size = gpu_params_vec.size();
    const auto current_device = c10::cuda::current_device();
    at::cuda::CUDAStream stream = at::cuda::getStreamFromExternal(reinterpret_cast<cudaStream_t>(stream_ptr), current_device);
    at::cuda::CUDAStreamGuard stream_guard(stream);
    for (int i = 0; i < size; i++)
    {
        auto gpu_parm = gpu_params_vec[i];
        gpu_params_vec[i].record_stream(stream);
        at::native::resize_bytes_cuda(gpu_params_vec[i].storage().unsafeGetStorageImpl(), 0);
    }
    nvtxRangePop();
}

std::vector<torch::Tensor> a2a_splits(std::vector<std::vector<int>> gpu_expert_assign, 
                                      std::vector<torch::Tensor> gather_list){
    std::vector<torch::Tensor> input_split_tensors;

    for (torch::Tensor output: gather_list){
        std::vector<torch::Tensor> _inner_sum;
        for (int dev_id = 0; dev_id < gpu_expert_assign.size(); dev_id++){
            std::vector<int> expert_assign = gpu_expert_assign[dev_id];
            torch::Tensor indices_tensor = torch::from_blob(expert_assign.data(), {static_cast<long>(expert_assign.size())}, torch::kInt);

            // torch::Tensor temptemp = output.to(indices_tensor.device()).index_select(0, indices_tensor);
            // torch::Tensor temptemp = 
            _inner_sum.push_back(output.to(indices_tensor).index_select(0, indices_tensor).sum());
        }
        input_split_tensors.push_back(torch::stack(_inner_sum, 0));
    }

    std::vector<torch::Tensor> a2a_split_vectors;
    a2a_split_vectors.push_back(torch::stack(input_split_tensors, 0));
    a2a_split_vectors.push_back(torch::transpose(a2a_split_vectors[0], 0, 1));
    return a2a_split_vectors;
}

std::vector<int> expert_splits(std::vector<std::vector<int>> gpu_expert_assign, 
                                                            std::vector<torch::Tensor> gather_list,
                                                            int rank){

    // torch::Tensor gather_list_stack = torch::stack(gather_list, 0);
    // torch::Tensor gather_list_sum = torch::sum(gather_list_stack, 0);


    // torch::Tensor indices_tensor = torch::from_blob(expert_assign.data(), {static_cast<long>(expert_assign.size())}, torch::kInt);
    // auto temp = gather_list_sum.index_select(0, indices_tensor);

    // float* data_ptr = temp.data_ptr<float>();
    // std::vector<int> expert_input_split(data_ptr, data_ptr + temp.numel());

    auto expert_assign = gpu_expert_assign[rank];
    std::vector<int> expert_output_split;
    for (auto exp_id : expert_assign){
        for(auto gate_output : gather_list){
            auto temp = gate_output[exp_id].item();
            expert_output_split.push_back(gate_output[exp_id].item().toInt());
        }
    }

    return expert_output_split;
}

torch::Tensor input_order(std::vector<int> gpu_expert_assign_flatten, torch::Tensor indices1_s){
    // std::vector<torch::Tensor> input_order_list;

    // for(int i = 0; i < gpu_expert_assign_flatten.size(); i++){
    //     // auto temp = (itorch::tensor(i, indices1_s.options()).expand_as(indices1_s) == indices1_s) * (i + 1);
    //     input_order_list.push_back((gpu_expert_assign_flatten[i] == indices1_s) * (i + 1));
    // } 

    // // std::cout << "input_order_list: " << input_order_list << std::endl;

    // auto stacked = torch::stack(input_order_list, 0);
    // auto summed = torch::sum(stacked, 0);
    // auto res = torch::argsort(summed, true);
    // return res;
    auto mapping = torch::zeros(indices1_s.sizes(), indices1_s.options().dtype(torch::kLong));
    for (int i = 0; i < gpu_expert_assign_flatten.size(); i++){
        // mapping[indices1_s == gpu_expert_assign_flatten[i]] = i;
        torch::Tensor indices = indices1_s == gpu_expert_assign_flatten[i];
        mapping.index_put_({indices}, i);
    }
    auto input_order_list = torch::argsort(mapping, true);
    return input_order_list;
}

torch::Tensor compensate_index_order(std::vector<int> gpu_expert_assign_flatten, torch::Tensor indices1_s, torch::Tensor counts){

    std::vector<torch::Tensor> compensate_index_list;

    for(int i = 0; i < gpu_expert_assign_flatten.size(); i++){
        compensate_index_list.push_back((gpu_expert_assign_flatten[i] == indices1_s) * counts[i]);
    }
    auto stacked = torch::stack(compensate_index_list);
    auto summed = torch::sum(stacked, 0);
    return summed;
}

// void put_task(std::string task, 
//               std::vector<std::vector<int>> gpu_expert_assign,
//               std::vector<int> gpu_expert_assign_flatten,
//               std::vector<torch::Tensor> gather_list,
//               torch::Tensor indices1_s,
//               int rank){

//   auto &engine = EsMoeEngine::getInstance();

//   if (task == "a2a"){
//     engine.a2a_future = engine.thread_pool_.enqueue(
//           a2a_splits, gpu_expert_assign, gather_list
//         );
//   }
//   else if (task == "expert"){
//     engine.expert_future = engine.thread_pool_.enqueue(
//           expert_splits, gpu_expert_assign, gather_list, rank
//         );

//   } else if (task == "input"){
//     engine.input_future = engine.thread_pool_.enqueue(
//           input_order, gpu_expert_assign_flatten, indices1_s
//         );
//   }
// }

// std::vector<torch::Tensor> get_a2a_splits(){
//   auto &engine = EsMoeEngine::getInstance();

//   auto res = engine.a2a_future.get();
//   return res;
// }

// std::pair<std::vector<int>, std::vector<int>> get_expert_splits(){
//   auto &engine = EsMoeEngine::getInstance();

//   auto res = engine.expert_future.get();
//   return res;
// }

// torch::Tensor get_input_order(){
//   auto &engine = EsMoeEngine::getInstance();

//   auto res = engine.input_future.get();
//   return res;
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("shared_pinned_memory", &shared_pinned_memory, "Get Shared and Pinned Tensor");

  m.def("upload_params", &upload_params, "Uploading params to GPU");
  m.def("upload_tensor", &upload_tensor, "Upload tensor in C++");
  m.def("offload_tensor", &offload_tensor, "Upload tensor in C++");

  m.def("upload_experts_params", &upload_experts_params, "Uploading experts params to GPU");

  m.def("offload_grads", &offload_grads, "Offloading grads to CPU");

  m.def("offload_experts_grads", &offload_experts_grads, "Offloading experts grads to CPU");

  // FREE PARAMS
  m.def("free_params", &free_params, "Free params in C++");

  // Gate functions
  m.def("a2a_splits", &a2a_splits, "A2A Splits");

  m.def("expert_splits", &expert_splits, "Experts Splits");

  m.def("input_order", &input_order, "Input Order");

  m.def("compensate_index_order", &compensate_index_order, "Compensate Post index order");

  //ThreadPool of C++ task
//   m.def("put_task", &put_task, "give task to C++ threadpool");

//   m.def("get_a2a_splits", &get_a2a_splits, "get_a2a_splits");

//   m.def("get_expert_splits", &get_expert_splits, "get_expert_splits");

//   m.def("get_input_order", &get_input_order, "get_input_order");

  // Engine Creation
  m.def("initialize_engine", []()
        {
            auto &engine = EsMoeEngine::getInstance();
        });
}