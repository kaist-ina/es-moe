#ifndef _GLOBAL_SEGMENT_MANAGER_H_
#define _GLOBAL_SEGMENT_MANAGER_H_

#include "common.h"
#include "shm_allocator.h"
#include "shm_message_queue.h"
#include "global_segment_manager_client.h"
#include <string>
#include <thread>
#include <memory>
#include <vector>
#include <set>
#include <tuple>
#include <queue>

struct MmapSegmentInfo {
    uint8_t *ptr_ = nullptr;
    uint8_t *ssd_ptr_ = nullptr;
    size_t size_ = 0;
    int fd_ = 0;

    MmapSegmentInfo() = default;   
    MmapSegmentInfo(uint8_t *ptr, size_t size, int shm_fd) : ptr_(ptr), ssd_ptr_(nullptr), size_(size), fd_(shm_fd) {}
};

class TrainProgressEstimator {
private: 
    mutable std::mutex mutex_;

    /* Heuristic for detecting number of segments per layer */
    std::map<layer_id_t, int> num_segment_per_layer_;
    size_t num_layers_ = 0;
    size_t num_experts_ = 0;


    layer_id_t current_layer_ = 0;
    expert_id_t current_expert_ = 0;
    bool current_forward_ = true;
    bool expect_flag_ = false;

public:
    void new_segment(const MemorySegmentKey key);
    inline size_t num_layers() const { return num_layers_; }
    inline size_t num_experts() const { return num_experts_; }

    void update(layer_id_t layer, expert_id_t expert, bool is_forward);
    inline bool is_time_to_expect() const {
        return expect_flag_;
    }

    std::map<MemorySegmentKey, int> expect_iter(const std::map<MemorySegmentKey, MmapSegmentInfo> &map_segments, size_t max) const;
};

class GlobalSegmentManager {

private:

    std::vector<ShmMessageQueue<ClientMessage>> vec_request_queue_;
    std::vector<ShmMessageQueue<ServerMessage>> vec_response_queue_;
    
    std::unique_ptr<ShmAllocator> shm_allocator_;
    std::unique_ptr<std::thread> message_handler_;

    std::mutex mutex_offload_;
    std::condition_variable cv_offload_;
    size_t num_max_ram_segments_ = 0;
    std::set<MemorySegmentKey> set_segments_;

    TrainProgressEstimator train_progress_estimator_;

    // std::set<MemorySegmentKey> set_loaded_segments_;
    // std::set<MemorySegmentKey> set_offloaded_segments_;
    // std::set<MemorySegmentKey> set_prefetching_segments_;
    // std::set<MemorySegmentKey> set_prefetched_segments_; // can be duplicated with set_loaded_segments_

    // std::set<MemorySegmentKey> set_offloading_segments_;
    // std::set<MemorySegmentKey> set_loading_segments_;
    std::mutex mutex_pending_ack_messages_;
    std::map<MemorySegmentKey, std::map<MemorySegmentLoadStatus::Status, std::vector<std::pair<int, ServerMessage>>>> map_pending_ack_messages_;

    std::mutex mutex_segment_mmaped_info_;
    std::map<MemorySegmentKey, MmapSegmentInfo> map_segment_mmaped_info_;
    

    std::string ssd_path_prefix_;

    bool is_initialized_ = false;
    bool is_finished_ = false;
    bool is_prefetch_enabled_ = true;
    bool prefetch_hint_first_ignore_ = true;

    std::map<MemorySegmentKey, int> map_prefetch_hint_;
    std::mutex mutex_prefetch_hint_;

    std::mutex mutex_hint_handler_;
    std::condition_variable cv_hint_handler_;
    std::queue<std::tuple<layer_id_t, expert_id_t, bool>> queue_hint_handler_;

    void initialize_shared_memory();
    void message_handler_main();


    void segment_loader_offloader_main_impl(bool is_offloading_handler);
    std::unique_ptr<std::thread> load_handler_;
    void segment_loader_main();
    std::unique_ptr<std::thread> offload_handler_;
    void segment_offloader_main();
    std::unique_ptr<std::thread> hint_handler_;
    void hint_handler_main();

    void inner_prefetch_impl();
    void inner_non_prefetch_impl();
    void inner_prefetch() { is_prefetch_enabled_ ? inner_prefetch_impl() : inner_non_prefetch_impl(); }
    
    MemorySegmentLoadStatus segment_status(MemorySegmentKey key) const;
    MemorySegmentLoadStatus &segment_status(MemorySegmentKey key);
    void set_segment_status(MemorySegmentKey key, MemorySegmentLoadStatus::Status status);

    void segment_register(MemorySegmentKey key, size_t req_qp_id, ClientMessage req); 
    void segment_acquire(MemorySegmentKey key, size_t req_qp_id, ClientMessage req); 
    void segment_release(MemorySegmentKey key, size_t req_qp_id, ClientMessage req);
    void prefetch_hint(layer_id_t layer, expert_id_t expert, bool is_forward);

    void register_segment_sync(MemorySegmentKey key, size_t size);
    void load_segment_sync(MemorySegmentKey key);
    void offload_segment_sync(MemorySegmentKey key);    

    void load_segment_enqueue_impl(MemorySegmentKey key, bool lock = true);
    void offload_segment_enqueue_impl(MemorySegmentKey key, bool lock = true);

    void process_ack(MemorySegmentKey key);

    void debug_print_segment_stat() const;

    

public:
    GlobalSegmentManager();
    ~GlobalSegmentManager();

};

#endif