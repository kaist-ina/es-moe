#ifndef _ENGINE_SHM_MANAGER_H
#define _ENGINE_SHM_MANAGER_H

#include <unistd.h>
#include <semaphore.h>
#include <cstring>
#include <cstdint>
#include <string>
#include <set>
#include <assert.h>

#define assert_p(expr) assert(expr)
#define UNUSED_VAR     __attribute__ ((unused))

static const std::string SHM_NAME_PREFIX("/esmoe-shm-");
static const std::string SEM_NAME_PREFIX("/esmoe-sem-");
static const off64_t META_BYTES_ALLOC = 64 * 1024 * 1024;
static const size_t MSG_RINGBUFFER_SIZE = 1024;
static const size_t MAX_LAYER = 256;
static const size_t MAX_EXPERT = 256;
static const size_t MAX_ORDER = 8;
static const size_t MAX_TYPE = 8;
static const size_t MAX_PENDING_DEPENDENCIES = 4096;
static const size_t MAX_NUM_SLAVE = 256;


enum class EventType {
    INVALID = 0,
    MASTER_DEP_ADDED = 1,
    MASTER_DEP_REMOVED = 2,
    MASTER_OFFLOADED = 3,
    MASTER_LOADED = 4,
};

UNUSED_VAR static const char *EventTypeStr[] = {
    "INVALID",
    "MASTER_DEP_ADDED",
    "MASTER_DEP_REMOVED",
    "MASTER_OFFLOADED",
    "MASTER_LOADED",
};

class ShmEvent {
public:
    int layer;
    int expert;
    int order;
    int type;
    EventType event_type;

    ShmEvent(int layer, int expert, int order, int type, EventType event_type) :
        layer(layer), expert(expert), order(order), type(type), event_type(event_type) {}
    ShmEvent(int layer, int expert, EventType event_type) :
        layer(layer), expert(expert), order(-1), type(-1), event_type(event_type) {}
};

enum MessageOperationType {
    FORWARD = 0,
    BACKWARD = 1,
    OPTIMIZATION = 2,
    CONFIRM = 3,
    SUBSCRIBE = 4,
};

enum MessageEventType {
    SCHEDULED = 0,
    STARTED = 1,
    FINISHED = 2,
    SKIPPED = 3,
};

enum MemorySegmentStatus {
    INVALID = 0, 
    ON_RAM = 1,
    ON_SSD = 2,
    PENDING = 3,
    ON_RAM_WAIT_CONFIRM = 4,
    ON_SSD_WAIT_CONFIRM = 5,
};

static const char * MemorySegmentStatusStr [] = {
    "INVALID", 
    "ON_RAM", 
    "ON_SSD", 
    "PENDING", 
    "ON_RAM_WAIT_CONFIRM", 
    "ON_SSD_WAIT_CONFIRM", 
};


enum class DependencyType {
    FORWARD,
    BACKWARD,
    OPTIMIZATION,
};


static const char *  MessageOperationTypeStr [] = {
    "FORWARD",
    "BACKWARD",
    "OPTIMIZATION",
};

static const char *  MessageEventTypeStr [] = {
    "SCHEDULED",
    "STARTED",
    "FINISHED",
};


struct alignas(4) MessageEntry {
    bool valid;
    int layer_id;
    int rank_from;
    int expert;
    int iter;
    MessageOperationType op_type;
    MessageEventType event;
    int data;
};


struct alignas(4) MemoryPoolMetaHeader {
    pthread_barrier_t barrier;
    pthread_mutex_t mutex_;
    pthread_cond_t cond_;

    pthread_mutex_t mutex_status_;
    pthread_cond_t cond_status_;


    pthread_mutex_t mutex_propagate_;
    pthread_cond_t cond_propagate_;

    bool barrier_init;
    uint32_t msg_ringbuffer_start_;
    uint32_t msg_ringbuffer_end_;
    uint32_t msg_ringbuffer_size_;
    MessageEntry msg_ringbuffer_[MSG_RINGBUFFER_SIZE];

    ShmEvent event_ringbuffer_[MAX_NUM_SLAVE][MSG_RINGBUFFER_SIZE];
    uint32_t event_ringbuffer_start_[MAX_NUM_SLAVE];
    uint32_t event_ringbuffer_end_[MAX_NUM_SLAVE];
    uint32_t event_ringbuffer_size_[MAX_NUM_SLAVE];

    pthread_mutex_t event_ringbuffer_mutex_[MAX_NUM_SLAVE];
    pthread_cond_t event_ringbuffer_cond_[MAX_NUM_SLAVE];


    MemorySegmentStatus offload_status_[MAX_LAYER][MAX_EXPERT][MAX_ORDER][MAX_TYPE];
    
    int opt_dependency_status_[MAX_PENDING_DEPENDENCIES];
    int fwd_dependency_status_[MAX_PENDING_DEPENDENCIES];
    int bwd_dependency_status_[MAX_PENDING_DEPENDENCIES];
    size_t num_slave_;
    int slave_pids_[MAX_NUM_SLAVE];
};


class ShmManager {

private:
    uint8_t *shm_meta_ptr_;
    sem_t* shm_meta_semaphore_;
    bool is_master_;
    pid_t local_session_id_;
    int local_rank_;
    bool is_finished_;

    void init_shared_memory_master();
    void init_shared_memory_slave();
    void lock();
    void unlock();

public:
    ShmManager(bool is_master = true, pid_t master_pid = 0, int local_rank = 0) : 
        shm_meta_ptr_(nullptr), shm_meta_semaphore_(nullptr), is_master_(false), 
        local_session_id_(0), local_rank_(local_rank), is_finished_(false) {
        local_session_id_ = master_pid;
        if (is_master || master_pid == 0) {
            is_master_ = true;
            init_shared_memory_master();
        } else {
            is_master_ = false;
            init_shared_memory_slave();
        }
    }

    ~ShmManager();

    inline MemoryPoolMetaHeader * meta_header() const { return reinterpret_cast<MemoryPoolMetaHeader *>(shm_meta_ptr_); }
    size_t get_num_slaves();
    void send_message(int layer_id, int rank_from, int expert, int iter, MessageOperationType op_type, MessageEventType event, int data = 0);
    MessageEntry recv_message(bool async);

    void master_report_offload_status(int layer_id, int expert, int order, int type, MemorySegmentStatus status);
    MemorySegmentStatus get_offload_status(int layer_id, int expert, int order, int type);
    void wait_until_offload_status(int layer_id, int expert, int order, int type, MemorySegmentStatus status);

    void mark_dependency(DependencyType dep_type, int iter, int layer_id, int expert);
    void unmark_dependency(DependencyType dep_type, int iter, int layer_id, int expert);
    std::set<int> get_dependency_set(DependencyType dep_type);
    
    void barrier(size_t size);

    void finish();
    void wake_other_ranks();
    bool sleep_for_wake();


    int add_slave(int pid);
    ShmEvent recv_slave_event(int slave_id);
    void push_slave_event(int slave_id, ShmEvent event);
    void broadcast_slave_event(ShmEvent event);


    static int bind_layer_expert (int layer, int expert) {
        return layer * 1000 + expert;
    }

    static int extract_layer_from_bind (int bound) {
        return (bound / 1000) % 1000;
    }

    static int extract_expert_from_bind (int bound) {
        return bound % 1000;
    }
    
};
#endif
