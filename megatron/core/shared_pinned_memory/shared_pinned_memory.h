#ifndef _SHARED_PINNED_MEMORY_H_
#define _SHARED_PINNED_MEMORY_H_

#include <thread>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <map>
#include <vector>
#include <array>
#include "shm_manager.h"

#define KEY_NUM 1000000

typedef uint32_t segid_t;

enum MemorySegmentType {
    PARAMETER = 1,
    GRADIENT = 2,
    OPTIMIZER = 3,
    OPTIMIZER_AUX = 4,
    OPTIMIZER_AUX2 = 5,
    OPTIMIZER_AUX3 = 6,
};

class MemorySegmentKey {
public:
    int layer;
    int expert;
    int order;
    int type;

    uint64_t hash() const {
        return (layer * 10000 + expert * 100 + order) * 100 + type;
    }

    bool operator==(const MemorySegmentKey &other) const
    { return hash() == other.hash(); }
    
    bool operator<(const MemorySegmentKey &other) const
    { return hash() < other.hash(); }

    MemorySegmentKey() = default;

    MemorySegmentKey(int layer = -1, int expert = -1, int order = -1, int type = -1) :
        layer(layer), expert(expert), order(order), type(type) {}
};

template <>
struct std::hash<MemorySegmentKey>
{
  std::size_t operator()(const MemorySegmentKey& k) const {
    return std::hash<uint64_t>()(k.hash());
  }
};


class MemorySegment {
public:
    int rank;
    int layer;
    int expert;
    int order;
    int type;
    int fd;
    bool pin;
    uint8_t *ptr;
    uint8_t *ssd_mmap_ptr;
    size_t size;
    std::string path;

    bool currently_pinned;

    std::unique_ptr<std::mutex> mutex;

    ShmEvent make_event(EventType event_type) const;

    MemorySegmentKey key() const {
        return MemorySegmentKey(layer, expert, order, type);
    }
};


struct MemorySegmentMasterState {
    MemorySegment *shm_ptr = nullptr;
    std::mutex mutex;
    std::set<int> subscribers;
    std::set<int> subscribers_pending_confirmation;
};

class MemorySegmentManager {
    std::map<uint8_t *, MemorySegment> map_segment_;
    std::array<std::set<int>, 3> set_layer_in_memory_; // param, grad, optimizer
    std::mutex mutex_;
    std::condition_variable cv_;

    std::map<MemorySegmentKey, MemorySegmentMasterState> map_segment_master_state_;
    std::mutex mutex_master_;

    bool ssd_offload_enable_ = false;
    bool ssd_offload_enable_init_ = false;
    size_t max_num_layer_in_memory_ = 0;

    const std::string SSD_PATH_PREFIX = "~/.cache/esmoe/";
    std::string ssd_path_prefix_;

    int rank_ = -1;
    size_t num_layers_ = 0;
    size_t num_experts_ = 0;
    int slave_id_ = -1;
    bool start_proc_ = false;    

    int current_layer_ = 0;
    std::map<int, int> num_segment_per_layer_;
    int current_num_segment_per_layer_ = 0;
    bool is_forward_ = true;
    bool is_master_ = false;
    bool is_finished_ = false;

    std::unique_ptr<ShmManager> shm_manager_;
    std::unique_ptr<std::thread> shm_manager_thread_;
    std::unique_ptr<std::thread> shm_manager_slave_thread_;
    std::unique_ptr<std::thread> shm_manager_offload_handler_thread_;

    
    std::unique_ptr<std::thread> master_offloader_thread_;
    std::unique_ptr<std::thread> cuda_event_synchronizer_thread_;
    std::unique_ptr<std::thread> forward_dependency_resolver_thread_;


    std::condition_variable cv_offload_handler_;
    std::mutex mutex_offload_handler_;
    int num_offload_handler_req_ = 0;

    void master_event_handler_main();
    void slave_event_handler_main();
    void offload_event_handler_main();
    void cuda_event_synchronizer_main();
    void master_offloader_main();

    std::set<cudaEvent_t> cuda_event_set_;
    std::condition_variable cv_cuda_event_;
    std::mutex mutex_cuda_event_;

    std::set<cudaEvent_t> cuda_event_finished_set_;
    std::condition_variable cv_cuda_event_finished_;
    std::mutex mutex_cuda_event_finished_;

    std::mutex mutex_pending_forward_dependencies_;
    std::map<int, cudaEvent_t> map_pending_forward_dependencies_;

    std::map<MemorySegmentKey, std::pair<MemorySegmentStatus, uint8_t *>> map_offload_queue_;
    std::mutex mutex_offload_queue_;
    std::condition_variable cv_offload_queue_;


    // MemorySegment &find_segment(int layer, int expert, int order, int type);



    void offload_to_ssd_master(uint8_t *ptr);
    void load_from_ssd_master(uint8_t *ptr);

    void _offload_to_ssd_master_internal(uint8_t *ptr);
    void _load_from_ssd_master_internal(uint8_t *ptr);

    

    void offload_to_ssd_slave(int layer, int expert, int order, int type);
    void load_from_ssd_slave(int layer, int expert, int order, int type);

    void subscribe_slave(int rank, int layer, int expert, int order, int type);

    void offload_logic_internal();
    void offload_logic_initial();
    void trigger_offload_logic();
    
    void mark_dependency(int layer, int expert, DependencyType dep_type);
    void unmark_dependency(int layer, int expert, DependencyType dep_type);

    void send_event_from_master(ShmEvent event, int slave_id = -1);
    void send_event_to_master(ShmEvent event);


    void debug_print_segment_stat();
    std::vector<int> static_experts_;
    

public:
    MemorySegmentManager();
    ~MemorySegmentManager();

    inline void set_rank(int rank) {
        if (rank == -1)
            rank_ = rank;
    }

    inline int rank() const { return rank_; }

    inline void indicate_master() {
        is_master_ = true;
    }

    uint8_t *alloc_pinned(size_t size, int rank, int layer, int expert, int order, int type, bool pinning = true);

    void report_upload(uint8_t *ptr);
    void report_optim_queued(int layer, int expert, int num_iter);
    void report_optim_started(int layer, int expert, int num_iter);
    void report_optim_finished(int layer, int expert, int num_iter);
    void report_optim_skipped(int layer, int expert, int num_iter);
    void report_post_forward(int layer, int expert, cudaStream_t stream);

    void report_static_experts(int layer, std::vector<int> experts);
    void wait_until_uploaded(uint8_t *ptr);
    void wait_until_uploaded(MemorySegment &segment);
    void wait_until_optim_uploaded(int layer, int expert);

    void check_in_cpu(uint8_t *ptr);

    void enable_ssd_offload(bool enable, int max_num_layer_in_memory = 0);

    void send_message(int layer_id, int rank_from, int expert, int iter, MessageOperationType op_type, MessageEventType event, int data = 0);

    void cuda_event_synchronize(cudaEvent_t event);
};



#endif