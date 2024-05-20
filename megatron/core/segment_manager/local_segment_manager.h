#ifndef _LOCAL_SEGMENT_MANAGER_H_
#define _LOCAL_SEGMENT_MANAGER_H_

#include "common.h"
#include "global_segment_manager.h"
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <thread>
#include <map>
#include <mutex>
#include <condition_variable>
#include <deque>



class LocalSegmentManager {

private:
    const std::string SSD_PATH_PREFIX = "~/.cache/esmoe/";
    std::map<MemorySegmentKey, MemorySegment> map_segment_;
    std::map<void *, MemorySegmentKey> map_segment_key_;
    std::mutex mutex_map_segment_;
    
    int rank_ = -1;
    bool is_finished_ = false;
    bool is_initialized_ = false;

    std::unique_ptr<std::thread> segment_release_worker_;
    std::condition_variable cv_segment_release_worker_;
    std::mutex mutex_segment_release_worker_;
    std::deque<MemorySegmentKey> queue_segment_release_;
    void segment_release_worker_main();

    std::unique_ptr<std::thread> cuda_release_worker_;
    std::condition_variable cv_cuda_release_worker_;
    std::mutex mutex_cuda_release_worker_;
    std::set<std::pair<cudaEvent_t, MemorySegmentKey>> set_cuda_release_;
    void cuda_release_worker_main();

    std::unique_ptr<GlobalSegmentManagerClient> global_segment_manager_;
    std::unique_ptr<GlobalSegmentManager> global_segment_manager_server_;

    /** acquire a segment, blocking call */
    void local_segment_acquire(MemorySegmentKey key); 
    
    /** release a segment, non-blocking call */
    void local_segment_release(MemorySegmentKey key);
    
    /** release a segment, non-blocking call */
    void local_segment_release(MemorySegmentKey key, cudaStream_t stream);

    void connect_global_segment_manager();
    void create_ssd_offload_path();
    
    /** Mark memory segment as CUDA pinned memory. */
    void cuda_register_segment(MemorySegment &segment);
    
    /** Unmark memory segment as CUDA pinned memory. */
    void cuda_unregister_segment(MemorySegment &segment);

    
    uint8_t *alloc_pinned(MemorySegmentKey key, size_t size, bool pinning);

    void hook_impl(layer_id_t layer, expert_id_t expert, cudaStream_t stream, bool acquire, std::function<bool(const MemorySegment &segment)> filter_cond);

public:
    LocalSegmentManager();
    ~LocalSegmentManager();

    /* Callable from Python */
    static LocalSegmentManager &getInstance() {
        static LocalSegmentManager instance;
        return instance;
    }

    inline bool is_initialized() const { return is_initialized_; }
    void initialize(int rank, std::unordered_map<std::string, int> options = {}, bool force = false);

    void pre_forward_hook(layer_id_t layer, expert_id_t expert);
    void post_forward_hook(layer_id_t layer, expert_id_t expert, cudaStream_t stream);

    void pre_backward_hook(layer_id_t layer, expert_id_t expert);
    void post_backward_hook(layer_id_t layer, expert_id_t expert, cudaStream_t stream);

    void pre_optimize_hook(layer_id_t layer, expert_id_t expert);
    void post_optimize_hook(layer_id_t layer, expert_id_t expert);

    torch::Tensor shared_pinned_memory(torch::Tensor tensor, int rank, int layer, int expert, int order, int type, bool pinning);

};


#endif