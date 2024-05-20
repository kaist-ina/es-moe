#ifndef _SEGMENT_MANAGER_COMMON_H_
#define _SEGMENT_MANAGER_COMMON_H_

#include <string>
#include <mutex>
#include <memory>
#include <csignal>
#include <cassert>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include "shm_message_queue.h"

#define PRINT_CPU_SSD_COPY 1
#define PRINT_STATE_TRANSITION 0
#define PRINT_PYTHON_API_CALL 0

/* Due to god damn KISTI */
#define LAZY_MAP_SSD_TO_REDUCE_NO_MMAP 0



#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

#define CHECK_ZERO(x)                                                   \
    if ((x)) {                                                         \
        char buf[2048];                                                 \
        snprintf (buf, 2048, "Assertion failed in \"%s\", line %d\n",   \
                 __FILE__, __LINE__);                                   \
        fflush (stdout);                                                \
        abort();                                                    \
    }                                                                   \
    else   // This 'else' exists to catch the user's following semicolon
    


typedef int layer_id_t;
typedef int expert_id_t;
typedef int order_id_t;
typedef int type_id_t;

static const layer_id_t MAX_NUM_LAYER = 256;
static const expert_id_t MAX_NUM_EXPERT = 256;
static const order_id_t MAX_NUM_ORDER = 8;
static const type_id_t MAX_NUM_TYPE = 8;
static const size_t MAX_NUM_LOCAL_MANAGER = 256;
static const size_t MAX_PENDING_DEPENDENCIES = 4096;
static const size_t MAX_NUM_CLIENTS = 128;

const std::string DEFAULT_SSD_PATH_PREFIX = "~/.cache/esmoe/";
static const size_t DEFAULT_NUM_MAX_RAM_SEGMENTS = 256;
static bool DISABLE_SEGMENT_MANAGER = false;




struct MemorySegmentLoadStatus {
    enum class Status : uint8_t {
        INVALID = 0,
        LOADED = 1,
        OFFLOADED = 2,
        PENDING_LOAD = 3,
        PENDING_OFFLOAD = 4,
        BUSY = 5,
        // PENDING_OFFLOAD_AND_LOAD = 5,
        // PENDING_LOAD_AND_OFFLOAD = 6,
    };

    inline const static char * StatusString[] = {
        "INVALID",
        "LOADED",
        "OFFLOADED",
        "PENDING_LOAD",
        "PENDING_OFFLOAD",
        "BUSY",
        // "PENDING_OFFLOAD",
        // "PENDING_OFFLOAD_AND_LOAD",
    };

    Status status_ = Status::INVALID;
    pthread_mutex_t mutex_;
    uint16_t refcnt_ = 0;
    bool busy = false;
};

struct MemorySegmentKey {
    enum class Type: type_id_t {
        INVALID = 0,
        PARAMETER = 1,
        GRADIENT = 2,
        OPTIMIZER = 3,
        OPTIMIZER_AUX = 4,
        OPTIMIZER_AUX2 = 5,
        OPTIMIZER_AUX3 = 6,
    };
    
    inline const static char * TypeString[] = {
        "INVALID",
        "PARAMETER",
        "GRADIENT",
        "OPTIMIZER",
        "OPTIMIZER_AUX",
        "OPTIMIZER_AUX2",
        "OPTIMIZER_AUX3",
    };

    layer_id_t layer;
    expert_id_t expert;
    order_id_t order;
    Type type;

    uint64_t hash() const {
        uint64_t h = static_cast<uint64_t>(type);
        h = h * MAX_NUM_LAYER + static_cast<type_id_t>(layer);
        h = h * MAX_NUM_EXPERT + expert;
        h = h * MAX_NUM_ORDER + order;
        return h;
    }

    static constexpr unsigned long long RoundUpto10s(const size_t maxNum, size_t current = 1) {
        return current >= maxNum ? current : RoundUpto10s(maxNum, current * 10);
    }

    const std::string hash_human() const {
        uint64_t h = static_cast<uint64_t>(type);
        h = h * RoundUpto10s(MAX_NUM_LAYER) + static_cast<type_id_t>(layer);
        h = h * RoundUpto10s(MAX_NUM_EXPERT) + expert;
        h = h * RoundUpto10s(MAX_NUM_ORDER) + order;
        return std::to_string(h);
    }

    bool operator==(const MemorySegmentKey &other) const { 
        return hash() == other.hash(); 
    }
    
    bool operator<(const MemorySegmentKey &other) const { 
        return hash() < other.hash(); 
    }

    std::string to_string() const {
        // get hash in hex str
        std::stringstream hex_hash_strm;
        hex_hash_strm << std::hex << std::setw(8) << std::setfill('0') << hash();

        return std::string("<Segment 0x") + hex_hash_strm.str() + " Layer=" + std::to_string(layer) + 
            " Expert=" + std::to_string(expert) + 
            " Order=" + std::to_string(order) + 
            " Type=" + TypeString[static_cast<size_t>(type)] + ">";
    }

    MemorySegmentKey() : layer(-1), expert(-1), order(-1), type(Type::INVALID) {};

    MemorySegmentKey(layer_id_t layer, expert_id_t expert, order_id_t order, MemorySegmentKey::Type type) :
        layer(layer), expert(expert), order(order), type(type) {
        assert(layer < MAX_NUM_LAYER);
        assert(expert < MAX_NUM_EXPERT);
        assert(order < MAX_NUM_ORDER);
        assert(static_cast<type_id_t>(type) < MAX_NUM_TYPE);
    }
};

template <>
struct std::hash<MemorySegmentKey>
{
  std::size_t operator()(const MemorySegmentKey& k) const {
    return std::hash<uint64_t>()(k.hash());
  }
};

class MemorySegment {
private:
    MemorySegmentKey key_;
    size_t size_ = 0;
    uint8_t *ptr_ = nullptr;
    uint8_t *ssd_ptr_ = nullptr;
    bool pin_ = false;
    bool currently_pinned_ = false;
    int shm_fd_ = -1;
    std::unique_ptr<std::mutex> mutex_;

    uint32_t local_refcnt_;

public:
    enum MemorySegmentState {
        OK,
        PENDING,
    };

    MemorySegment() = default;
    MemorySegment(MemorySegmentKey key, size_t size, uint8_t *ptr, bool pin, bool currently_pinned, int shm_fd) : 
        key_(key), size_(size), ptr_(ptr), pin_(pin), currently_pinned_(currently_pinned), shm_fd_(shm_fd), mutex_(std::make_unique<std::mutex>()), local_refcnt_(0) {};
        
    
    MemorySegment(MemorySegment&&) = default;
    MemorySegment& operator=(MemorySegment&&) = default;

    MemorySegment(const MemorySegment &) = delete;
    MemorySegment& operator=(const MemorySegment &) = delete;

    inline layer_id_t layer() const { return key_.layer; }
    inline expert_id_t expert() const { return key_.expert; }
    inline order_id_t order() const { return key_.order; }
    inline MemorySegmentKey::Type type() const { return key_.type; }
    inline size_t size() const { return size_; }
    inline uint8_t *ptr() const { return ptr_; }
    inline uint8_t *ssd_ptr() const { return ssd_ptr_; }
    inline std::string path() const { return key_.hash_human(); }
    inline bool pin() const { return pin_; }
    inline bool &currently_pinned() { return currently_pinned_; }
    inline const MemorySegmentKey &key() const { return key_; }
    inline std::mutex &mutex() { return *mutex_; }
    inline uint32_t &local_refcnt() { return local_refcnt_; }
};


struct ClientMessage {
    enum class Type {
        INVALID,
        REGISTER,
        ACQUIRE,
        RELEASE,
        HINT,
    };

    inline const static char * TypeString[] = {
        "INVALID",
        "REGISTER",
        "ACQUIRE",
        "RELEASE",
        "HINT",
    };

    inline static std::mutex global_mutex_;

    static int get_next_message_id() {
        std::unique_lock<std::mutex> lock(global_mutex_);
        static int message_id_ = 0;
        return ++message_id_;
    }
    

    int seq_ = -1;
    bool valid_ = false;
    Type type_ = Type::INVALID;

    MemorySegmentKey key_;
    size_t size_ = 0;

    layer_id_t layer_ = 0;
    expert_id_t expert_ = 0;
    bool forward_ = false;

    ClientMessage() = default;
    ClientMessage(Type type, MemorySegmentKey key) : seq_(get_next_message_id()), valid_(true), type_(type), key_(key) {}
    ClientMessage(Type type, MemorySegmentKey key, size_t size) : seq_(get_next_message_id()), valid_(true), type_(type), key_(key), size_(size) {}
    ClientMessage(layer_id_t layer, expert_id_t expert, bool forward) : seq_(get_next_message_id()), valid_(true), type_(Type::HINT), layer_(layer), expert_(expert), forward_(forward) {}
};

struct ServerMessage {
    
    enum class Type {
        INVALID = 0,
        REGISTER_ACK,
        ACQUIRE_ACK,
    };

    inline const static char * TypeString[] = {
        "INVALID",
        "REGISTER_ACK",
        "ACQUIRE_ACK",
    };

    inline static std::mutex global_mutex_;

    static int get_next_message_id() {
        std::unique_lock<std::mutex> lock(global_mutex_);
        static int message_id_ = 0;
        return ++message_id_;
    }
    
    int seq_ = -1;
    int ack_ = -1;
    bool valid_ = false;
    Type type_ = Type::INVALID;

    std::string to_string() const {
        return std::string("[") + TypeString[static_cast<size_t>(type_)] + "] ACK=" + std::to_string(ack_) + " Valid=" + std::to_string(valid_);
    }
    
    ServerMessage() = default;
    ServerMessage(Type type, int ack, MemorySegmentKey key) : seq_(get_next_message_id()), ack_(ack), valid_(true), type_(type) {}
};

struct alignas(4) SharedMemoryHeader {
    pthread_mutex_t mutex_;
    pthread_cond_t cond_;
    
    size_t num_clients_;
    MessageQueuePair<ClientMessage, ServerMessage> request_qp_[MAX_NUM_CLIENTS];

    MemorySegmentLoadStatus offload_status_[MAX_NUM_LAYER][MAX_NUM_EXPERT][MAX_NUM_ORDER][MAX_NUM_TYPE];
};


[[maybe_unused]] static void initialize_thread(std::string thread_name) {
    sigset_t mask;
    sigemptyset (&mask);
    sigaddset (&mask, SIGTERM);
    sigaddset (&mask, SIGHUP);
    pthread_sigmask(SIG_BLOCK, &mask, NULL);
    
    char buffer[16];
    snprintf(buffer, 16, "%s", thread_name.c_str());
    pthread_setname_np(pthread_self(), buffer);
}


static inline size_t is_disable_ssd_offload() {
    
    static bool initialized = false;
    static bool result = false;
    if (!initialized) {
        
        if (DISABLE_SEGMENT_MANAGER) {
            result = true;
        } else if (const char* env_p = std::getenv("DISABLE_SSD_OFFLOAD")) {
            int parsed = std::stoi(std::string(env_p));
            result = parsed > 0;
        }

        if (result) {
            std::cerr << "Turing off SSD offload" << std::endl;
        } else {
            std::cerr << "Turing on SSD offload" << std::endl;
        }
        initialized = true;
    }
    return result;
}

static inline size_t get_num_max_ram_segments() {

    static bool initialized = false;
    static size_t result = 0;
    if (!initialized) {
        if (const char* env_p = std::getenv("NUM_RAM_MAX_SEGMENTS")) {
            size_t parsed = std::stoul(std::string(env_p));
            if (parsed > 0) {
                result = parsed;
            } else {
                result = DEFAULT_NUM_MAX_RAM_SEGMENTS;
            }
        } else {
            result = DEFAULT_NUM_MAX_RAM_SEGMENTS;
        }
        std::cerr << "# of max RAM segment is set to " << result << std::endl;
        initialized = true;
    }
    return result;
}


static inline std::string get_ssd_path_prefix() {

    static bool initialized = false;
    static std::string result;
    if (!initialized) {
        if (const char* env_p = std::getenv("SSD_PATH_PREFIX")) {
            std::string path(env_p);
            if (path.size() > 0) {
                result = path;
                if (path.end()[-1] != '/') {
                    result += "/";
                }
            } else {
                result = DEFAULT_SSD_PATH_PREFIX;
            }
        } else {
            result = DEFAULT_SSD_PATH_PREFIX;   
        }
        std::cerr << "SSD offload path is set to " << result << std::endl;
        initialized = true;
    }
    return result;
}


#endif
