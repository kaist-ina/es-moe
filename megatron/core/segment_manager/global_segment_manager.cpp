#include "global_segment_manager.h"
#include <cassert>
#include <cstdio>
#include <vector>
#include <iostream>
#include <sstream>
#include <queue>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <wordexp.h>


// std::mutex ServerMessage::global_mutex_;
// std::mutex ClientMessage::global_mutex_;


GlobalSegmentManager::GlobalSegmentManager() {
    fprintf(stderr, "[%d] GlobalSegmentManager::GlobalSegmentManager\n", getpid());
    num_max_ram_segments_ = get_num_max_ram_segments();

    initialize_shared_memory();

    /* expand tilde */
    wordexp_t exp_result;
    wordexp(get_ssd_path_prefix().c_str(), &exp_result, 0);
    ssd_path_prefix_ = std::string(exp_result.we_wordv[0]);
    wordfree(&exp_result);

    /* check dir exists */
    struct stat dir;
    if (stat(ssd_path_prefix_.c_str(), &dir) != 0) {
        perror("stat");
        abort();
    }

    if (!S_ISDIR(dir.st_mode)) {
        std::cerr << "Path " << ssd_path_prefix_ << "does not exist." << std::endl;
        abort();
    }

    message_handler_ = std::make_unique<std::thread>([this] { message_handler_main(); });
    offload_handler_ = std::make_unique<std::thread>([this] { segment_offloader_main(); });
    load_handler_ = std::make_unique<std::thread>([this] { segment_loader_main(); });
    hint_handler_ = std::make_unique<std::thread>([this] { hint_handler_main(); });
}

GlobalSegmentManager::~GlobalSegmentManager() {
    is_finished_ = true;

    if (shm_allocator_) {
        auto header = reinterpret_cast<SharedMemoryHeader *>(shm_allocator_->shm_ptr());
        pthread_cond_broadcast(&header->cond_);
    }

    cv_offload_.notify_all();
    if (message_handler_) {
        fprintf(stderr, "GlobalSegmentManager::~GlobalSegmentManager: Waiting for message_handler_ to join\n");
        message_handler_->join();
    }
    if (offload_handler_) {
        fprintf(stderr, "GlobalSegmentManager::~GlobalSegmentManager: Waiting for offload_handler_ to join\n");
        offload_handler_->join();
    }
    if (load_handler_) {
        fprintf(stderr, "GlobalSegmentManager::~GlobalSegmentManager: Waiting for load_handler_ to join\n");
        load_handler_->join();
    }

    if (hint_handler_) {
        cv_hint_handler_.notify_all();
        fprintf(stderr, "GlobalSegmentManager::~GlobalSegmentManager: Waiting for hint_handler_ to join\n");
        hint_handler_->join();
    }
    
    fprintf(stderr, "GlobalSegmentManager::~GlobalSegmentManager: Waiting for munmap\n");
    {
        std::unique_lock<std::mutex> ul(mutex_offload_);
        for (auto &pair : map_segment_mmaped_info_) {
            auto &info = pair.second;
            if (info.ptr_)
                munmap(info.ptr_, info.size_);
            if (info.ssd_ptr_)
                munmap(info.ssd_ptr_, info.size_);
        }
    }

    shm_allocator_ = nullptr;
    
    fprintf(stderr, "GlobalSegmentManager::~GlobalSegmentManager: Terminated\n");
}

void GlobalSegmentManager::initialize_shared_memory() {
    fprintf(stderr, "Initializing SHM Master, pgid=%d\n", getpgid(0));
    shm_allocator_ = std::make_unique<ShmAllocator>(true, getpgid(0), [this] (uint8_t *ptr) {
        auto header = reinterpret_cast<SharedMemoryHeader *>(ptr);
        int ret;

        pthread_mutexattr_t mutex_attr;
        memset(&mutex_attr, 0, sizeof(pthread_mutexattr_t));
        ret = pthread_mutexattr_setpshared(&mutex_attr, PTHREAD_PROCESS_SHARED);
        assert(ret == 0);
        ret = pthread_mutex_init(&header->mutex_, &mutex_attr);
        assert(ret == 0);

        pthread_condattr_t cond_attr;
        memset(&cond_attr, 0, sizeof(pthread_condattr_t));
        ret = pthread_condattr_setpshared(&cond_attr, PTHREAD_PROCESS_SHARED);
        assert(ret == 0);    
        ret = pthread_cond_init(&header->cond_, &cond_attr);
        assert(ret == 0);
    });
    
    is_initialized_ = true;
    fprintf(stderr, "Initializing SHM Master OK\n");
}


void GlobalSegmentManager::message_handler_main() {
    initialize_thread("GlblMsgHndlr");

    auto header = reinterpret_cast<SharedMemoryHeader *>(shm_allocator_->shm_ptr());
    bool need_wait = false;
    pthread_mutex_lock(&header->mutex_);
    while (!is_finished_) {
        if (need_wait) {
            pthread_cond_wait(&header->cond_, &header->mutex_);
        } else {
            need_wait = true;
        }

        while (vec_request_queue_.size() < header->num_clients_) {
            auto buffer = &header->request_qp_[vec_request_queue_.size()].uplink_;
            vec_request_queue_.emplace_back(buffer);
            assert(vec_request_queue_.back().initialized());
            // fprintf(stderr, "Created request queue %zu, buffer %p\n", vec_request_queue_.size(), buffer);
        }

        while (vec_response_queue_.size() < header->num_clients_) {
            auto buffer = &header->request_qp_[vec_response_queue_.size()].downlink_;
            vec_response_queue_.emplace_back(buffer);
            assert(vec_response_queue_.back().initialized());
            // fprintf(stderr, "Created response queue %zu, buffer %p\n", vec_response_queue_.size(), buffer);
        }

        assert(vec_request_queue_.size() == header->num_clients_);
        assert(vec_response_queue_.size() == header->num_clients_);

        for (size_t qp_id = 0; qp_id < header->num_clients_; qp_id++) {
            assert(vec_request_queue_[qp_id].initialized());

            if (vec_request_queue_[qp_id].size() == 0) {
                continue;
            }

            while (true) {
                ClientMessage request = vec_request_queue_[qp_id].pop(true);
                if (!request.valid_)
                    break;
                need_wait = false;
                // fprintf(stderr, "[%d] Received request from client %zu type=%s seq=%d %s\n", 
                //     getpid(), qp_id, ClientMessage::TypeString[static_cast<size_t>(request.type_)], request.seq_, request.key_.to_string().c_str());   

                // process here
                if (request.type_ == ClientMessage::Type::REGISTER) {
                    segment_register(request.key_, qp_id, request);
                } else if (request.type_ == ClientMessage::Type::ACQUIRE) {
                    segment_acquire(request.key_, qp_id, request);
                } else if (request.type_ == ClientMessage::Type::RELEASE) {
                    segment_release(request.key_, qp_id, request);
                } else if (request.type_ == ClientMessage::Type::HINT) {
                    prefetch_hint(request.layer_, request.expert_, request.forward_);
                } else {
                    assert(false);
                }
            }
        }
    }

    pthread_mutex_unlock(&header->mutex_);
}

void GlobalSegmentManager::segment_register(MemorySegmentKey key, size_t req_qp_id, ClientMessage req) {
    auto header = reinterpret_cast<SharedMemoryHeader *>(shm_allocator_->shm_ptr());

    bool new_segment = false;
    std::unique_lock<std::mutex> ul(mutex_offload_);
    auto &status = header->offload_status_[key.layer][key.expert][key.order][static_cast<size_t>(key.type)];
    if (status.status_ == MemorySegmentLoadStatus::Status::INVALID) {
        status.status_ = MemorySegmentLoadStatus::Status::LOADED;
        new_segment = true;
        set_segments_.insert(key);

        int ret;
        pthread_mutexattr_t mutex_attr;
        memset(&mutex_attr, 0, sizeof(pthread_mutexattr_t));
        ret = pthread_mutexattr_setpshared(&mutex_attr, PTHREAD_PROCESS_SHARED);
        assert(ret == 0);
        ret = pthread_mutexattr_settype(&mutex_attr, PTHREAD_MUTEX_ERRORCHECK);
        assert(ret == 0);
        ret = pthread_mutex_init(&status.mutex_, &mutex_attr);
        assert(ret == 0);

        train_progress_estimator_.new_segment(key);
    }

    assert(req_qp_id < vec_request_queue_.size());
    vec_response_queue_[req_qp_id].push(ServerMessage(ServerMessage::Type::REGISTER_ACK, req.seq_, key));
    
    if (!new_segment)
        return;

    register_segment_sync(key, req.size_);
}

MemorySegmentLoadStatus &GlobalSegmentManager::segment_status(MemorySegmentKey key) {
    assert(shm_allocator_);
    SharedMemoryHeader *shm_header = reinterpret_cast<SharedMemoryHeader *>(shm_allocator_->shm_ptr());
    return shm_header->offload_status_[key.layer][key.expert][key.order][static_cast<size_t>(key.type)];
}

MemorySegmentLoadStatus GlobalSegmentManager::segment_status(MemorySegmentKey key) const {
    assert(shm_allocator_);
    SharedMemoryHeader *shm_header = reinterpret_cast<SharedMemoryHeader *>(shm_allocator_->shm_ptr());
    return shm_header->offload_status_[key.layer][key.expert][key.order][static_cast<size_t>(key.type)];
}

void GlobalSegmentManager::set_segment_status(MemorySegmentKey key, MemorySegmentLoadStatus::Status status) {
    assert(shm_allocator_);
    SharedMemoryHeader *shm_header = reinterpret_cast<SharedMemoryHeader *>(shm_allocator_->shm_ptr());
    shm_header->offload_status_[key.layer][key.expert][key.order][static_cast<size_t>(key.type)].status_ = status;
}


void GlobalSegmentManager::load_segment_enqueue_impl(MemorySegmentKey key, bool lock) {
    /* Assume already locked */
    auto &status = segment_status(key);
    if (lock) {
        CHECK_ZERO(pthread_mutex_lock(&status.mutex_));
    }
    auto s = status.status_;
    switch (status.status_) {
        case MemorySegmentLoadStatus::Status::LOADED:
            std::cerr << "WARN: Segment " << key.to_string() << " is already loaded, no-op" << std::endl;
            break;

        case MemorySegmentLoadStatus::Status::INVALID:
        case MemorySegmentLoadStatus::Status::BUSY:
            assert(false);
            break;

        case MemorySegmentLoadStatus::Status::PENDING_OFFLOAD:
        case MemorySegmentLoadStatus::Status::PENDING_LOAD:
        case MemorySegmentLoadStatus::Status::OFFLOADED:
            status.status_ = MemorySegmentLoadStatus::Status::PENDING_LOAD;
            break;
    }

    if (lock) {
        CHECK_ZERO(pthread_mutex_unlock(&status.mutex_));
    }
    cv_offload_.notify_all();
}


void GlobalSegmentManager::offload_segment_enqueue_impl(MemorySegmentKey key, bool lock) {

    auto &status = segment_status(key);
    if (lock) {
        CHECK_ZERO(pthread_mutex_lock(&status.mutex_));
    }

    switch (status.status_) {
        case MemorySegmentLoadStatus::Status::OFFLOADED:
            std::cerr << "WARN: Segment " << key.to_string() << " is already offloaded, no-op" << std::endl;
            break;
        case MemorySegmentLoadStatus::Status::INVALID:
        case MemorySegmentLoadStatus::Status::BUSY:
            assert(false);
            break;

        case MemorySegmentLoadStatus::Status::PENDING_LOAD:
        case MemorySegmentLoadStatus::Status::PENDING_OFFLOAD:
        case MemorySegmentLoadStatus::Status::LOADED:
            status.status_ = MemorySegmentLoadStatus::Status::PENDING_OFFLOAD;
            break;
    }

    if (lock) {
        CHECK_ZERO(pthread_mutex_unlock(&status.mutex_));
    }
    cv_offload_.notify_all();
}

void GlobalSegmentManager::register_segment_sync(MemorySegmentKey key, size_t size) {
    /* Assume already locked */

    const char* hash_human_cstr = key.hash_human().c_str();

    int shm_fd = shm_open(hash_human_cstr, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
    if (shm_fd == -1) {
        std::cerr << hash_human_cstr <<", Failed to create/open shared memory object: " << std::strerror(errno) << std::endl;
        abort();
    }

    if (ftruncate(shm_fd, size) == -1) {
        perror("ftruncate");
        abort();
    }

    uint8_t *memory_segment_ptr = reinterpret_cast<uint8_t *>(mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0));
    if (memory_segment_ptr == MAP_FAILED) {
        perror("mmap");
        abort();
    }

    std::unique_lock<std::mutex> ul(mutex_segment_mmaped_info_);
    map_segment_mmaped_info_[key] = MmapSegmentInfo(memory_segment_ptr, size, shm_fd);
}

void GlobalSegmentManager::segment_acquire(MemorySegmentKey key, size_t req_qp_id, ClientMessage req) {
    auto &status = segment_status(key);
    CHECK_ZERO(pthread_mutex_lock(&status.mutex_));
    assert(status.status_ != MemorySegmentLoadStatus::Status::INVALID);
    status.refcnt_++;

    bool already_loaded = false;
    if (status.status_ == MemorySegmentLoadStatus::Status::LOADED) {
        /* Send ACK immediately */
        assert(req_qp_id < vec_request_queue_.size());
        vec_response_queue_[req_qp_id].push(ServerMessage(ServerMessage::Type::ACQUIRE_ACK, req.seq_, key));
        already_loaded = true;
    } else {
        // fprintf(stderr, "[%d] %s is not loaded, queuing message (%s)\n", getpid(), key.to_string().c_str(), MemorySegmentLoadStatus::StatusString[static_cast<size_t>(status.status_)]);
        {
            std::unique_lock<std::mutex> ul(mutex_pending_ack_messages_);
            map_pending_ack_messages_[key][MemorySegmentLoadStatus::Status::LOADED].emplace_back(
                req_qp_id, ServerMessage(ServerMessage::Type::ACQUIRE_ACK, req.seq_, key));   
        }
        load_segment_enqueue_impl(key, false);
    }

    CHECK_ZERO(pthread_mutex_unlock(&status.mutex_));

    if (already_loaded) {
        inner_prefetch();
    }

    cv_offload_.notify_all();
}

void GlobalSegmentManager::segment_release(MemorySegmentKey key, size_t req_qp_id, ClientMessage req) {
    auto &status = segment_status(key);
    CHECK_ZERO(pthread_mutex_lock(&status.mutex_));
    assert(status.status_ != MemorySegmentLoadStatus::Status::INVALID);
    assert(status.refcnt_ > 0);
    status.refcnt_--;
    // fprintf(stderr, "[%d] %s refcnt=%d\n", getpid(), key.to_string().c_str(), status.refcnt_);
    
    bool already_offloaded = false;
    if (status.status_ == MemorySegmentLoadStatus::Status::OFFLOADED) {
        /* Send ACK immediately */
        already_offloaded = true;
    }
    CHECK_ZERO(pthread_mutex_unlock(&status.mutex_));

    if (already_offloaded) {
        inner_prefetch();
    }
    cv_offload_.notify_all();
}


void GlobalSegmentManager::prefetch_hint(layer_id_t layer, expert_id_t expert, bool forward) {

    assert(layer < static_cast<layer_id_t>(train_progress_estimator_.num_layers()));
    assert(expert < static_cast<expert_id_t>(train_progress_estimator_.num_experts()));
    /* to prevent initial glitches */
    if (prefetch_hint_first_ignore_) {
        if (layer > 0)
            prefetch_hint_first_ignore_ = false;
        return;
    }

    std::unique_lock<std::mutex> ul(mutex_hint_handler_);
    queue_hint_handler_.push(std::tuple<layer_id_t, expert_id_t, bool>(layer, expert, forward));
    cv_hint_handler_.notify_one();
}

void GlobalSegmentManager::inner_non_prefetch_impl() {
    
    /* Evict overloaded segments */
    std::vector<MemorySegmentKey> keys_to_evict;
    {
        std::unique_lock<std::mutex> ul(mutex_offload_);
        size_t num_segments_in_memory = 0;
        for (auto it = set_segments_.begin(); it != set_segments_.end(); ++it) {
            auto &key = *it;
            auto &status = segment_status(key);
            if (status.status_ == MemorySegmentLoadStatus::Status::LOADED) {
                num_segments_in_memory++;
                if (num_segments_in_memory > num_max_ram_segments_) {
                    if (pthread_mutex_trylock(&status.mutex_) == 0) {
                        if (status.refcnt_ == 0) {
                            num_segments_in_memory--;
                            keys_to_evict.push_back(key);
                        } else {
                            CHECK_ZERO(pthread_mutex_unlock(&status.mutex_));
                        }
                    }
                }
            }
        }
    }

    for (auto &key : keys_to_evict) {
        auto &status = segment_status(key);
        // fprintf(stderr, "Evicting segment %s \n", key.to_string().c_str());
        offload_segment_enqueue_impl(key, false);
        CHECK_ZERO(pthread_mutex_unlock(&status.mutex_));
    }

    cv_offload_.notify_all();
}

void GlobalSegmentManager::inner_prefetch_impl() {

    if (map_prefetch_hint_.size() == 0) {
        inner_non_prefetch_impl();
        return;
    }
    
    /* Evict overloaded segments */

    std::unique_lock<std::mutex> ul(mutex_prefetch_hint_);
    std::map<MemorySegmentKey, int> map_prefetch_hint(map_prefetch_hint_);
    ul.unlock();

    struct customLess
    {
        bool operator()(const std::pair<int, MemorySegmentKey> &l, const std::pair<int, MemorySegmentKey> &r) const { return l > r; }
    } ;
    struct customGreater
    {
        bool operator()(const std::pair<int, MemorySegmentKey> &l, const std::pair<int, MemorySegmentKey> &r) const { return l < r; }
    } ;
    
    std::priority_queue<std::pair<int, MemorySegmentKey>, std::vector<std::pair<int, MemorySegmentKey>>, customGreater> pq_to_evict;
    std::priority_queue<std::pair<int, MemorySegmentKey>, std::vector<std::pair<int, MemorySegmentKey>>, customLess> pq_to_prefetch;


    std::vector<MemorySegmentKey> keys_to_evict;
    std::vector<MemorySegmentKey> keys_to_prefetch;
    size_t num_segments_in_memory = 0;
    size_t num_inevicatble_in_memory = 0;
    {
        std::unique_lock<std::mutex> ul(mutex_offload_);

        /* first pass: mark segment priority */
        for (auto it = set_segments_.begin(); it != set_segments_.end(); ++it) {
            auto &key = *it;
            auto &status = segment_status(key);
            if (status.status_ == MemorySegmentLoadStatus::Status::LOADED) {
                num_segments_in_memory++;
            } 

            if (status.refcnt_ == 0) {
                auto it = map_prefetch_hint.find(key);
                int priority = it != map_prefetch_hint.end() ? it->second : std::numeric_limits<int>::max();
                pq_to_evict.push(std::make_pair(priority, key));
                pq_to_prefetch.push(std::make_pair(priority, key));
            } else {
                if (status.status_ == MemorySegmentLoadStatus::Status::LOADED) {
                    num_inevicatble_in_memory++;
                } 
            }
        }
    }

    assert(num_segments_in_memory >= num_inevicatble_in_memory);
    std::set<MemorySegmentKey> discussed_keys;

    /* second pass: evict + prefetch using pq */
    while (true) {
        bool updated = false;

        if (num_segments_in_memory < num_max_ram_segments_ && pq_to_prefetch.size()) {
            updated = true;
            auto &key = pq_to_prefetch.top().second;
            if (discussed_keys.find(key) == discussed_keys.end()) {
                auto &status = segment_status(key);
                if (pthread_mutex_trylock(&status.mutex_) == 0) {
                    if (status.status_ == MemorySegmentLoadStatus::Status::OFFLOADED) {
                        num_segments_in_memory++;
                        status.status_ = MemorySegmentLoadStatus::Status::PENDING_LOAD;
                        keys_to_prefetch.push_back(key);
                        discussed_keys.insert(key);
                    } else {
                        CHECK_ZERO(pthread_mutex_unlock(&status.mutex_));
                    }
                }
            }
            pq_to_prefetch.pop();
        }

        if (num_segments_in_memory >= num_max_ram_segments_ && pq_to_evict.size()) {
            updated = true;
            auto &key = pq_to_evict.top().second;
            if (discussed_keys.find(key) == discussed_keys.end()) {
                auto &status = segment_status(key);
                if (pthread_mutex_trylock(&status.mutex_) == 0) {
                    if (status.refcnt_ == 0 && status.status_ == MemorySegmentLoadStatus::Status::LOADED) {
                        num_segments_in_memory--;
                        status.status_ = MemorySegmentLoadStatus::Status::PENDING_OFFLOAD;
                        keys_to_evict.push_back(key);
                        discussed_keys.insert(key);
                    } else {
                        CHECK_ZERO(pthread_mutex_unlock(&status.mutex_));
                    }
                }
            }
            pq_to_evict.pop();
        }

        if (!updated)
            break;
    }

#if PRINT_CPU_SSD_COPY    
    std::stringstream sstr0, sstr1;
    if (keys_to_evict.size()) {
        for (auto it = keys_to_evict.begin(); it != keys_to_evict.end(); ++it) {
            sstr0 << it->layer << "/" << it->expert << "/" << it->order << "/" << MemorySegmentKey::TypeString[static_cast<size_t>(it->type)][0] << " ";
        }
        fprintf(stderr, KGRN "[%d] Will Evict %s, num_segments_in_memory=%lu, num_inevicatble_in_memory=%lu\n" KNRM, 
            getpid(), sstr0.str().c_str(), num_segments_in_memory, num_inevicatble_in_memory);        
    }
    if (keys_to_prefetch.size()) {
        for (auto it = keys_to_prefetch.begin(); it != keys_to_prefetch.end(); ++it) {
            sstr1 << it->layer << "/" << it->expert << "/" << it->order << "/" << MemorySegmentKey::TypeString[static_cast<size_t>(it->type)][0] << " ";
        }
        fprintf(stderr, KGRN "[%d] Will Prefetch %s, num_segments_in_memory=%lu, num_inevicatble_in_memory=%lu\n" KNRM, 
            getpid(), sstr1.str().c_str(), num_segments_in_memory, num_inevicatble_in_memory);
    }
#endif

    for (auto &key : keys_to_evict) {
        auto &status = segment_status(key);
        offload_segment_enqueue_impl(key, false);
        CHECK_ZERO(pthread_mutex_unlock(&status.mutex_));
    }

    for (auto &key : keys_to_prefetch) {
        auto &status = segment_status(key);
        load_segment_enqueue_impl(key, false);
        CHECK_ZERO(pthread_mutex_unlock(&status.mutex_));
    }

    // if (keys_to_evict.size() || keys_to_prefetch.size())
    //     cv_offload_.notify_all();
}

void GlobalSegmentManager::process_ack(MemorySegmentKey key) {
    auto status = segment_status(key);

    std::unique_lock<std::mutex> ul(mutex_pending_ack_messages_);
    for (auto &msg : map_pending_ack_messages_[key][status.status_]) {
        /* Send ACK */
        assert(msg.first < static_cast<int>(vec_request_queue_.size()));
        vec_response_queue_[msg.first].push(msg.second);
    }
    map_pending_ack_messages_[key][status.status_].clear();        
}

void GlobalSegmentManager::segment_loader_offloader_main_impl(bool is_offloading_handler) {
    auto header = reinterpret_cast<SharedMemoryHeader *>(shm_allocator_->shm_ptr());
    MemorySegmentLoadStatus::Status target_status = is_offloading_handler ? MemorySegmentLoadStatus::Status::OFFLOADED : MemorySegmentLoadStatus::Status::LOADED;

    bool wait_next = false;
    while (!is_finished_) {
        std::vector<MemorySegmentKey> keys_to_process;

        {
            std::unique_lock<std::mutex> ul(mutex_offload_);

            if (wait_next)
                cv_offload_.wait(ul);
            
            for (auto it = set_segments_.begin(); it != set_segments_.end(); ++it) {
                auto &key = *it;
                auto &status = segment_status(key);
                auto current_status = status.status_;
                bool processed = false;

                auto trylock = pthread_mutex_trylock(&status.mutex_);
                if (trylock != 0) {
                    assert(trylock == EBUSY);
                    continue;
                }

                assert(status.status_ != MemorySegmentLoadStatus::Status::BUSY);

                if (is_offloading_handler) {
                    if (current_status == MemorySegmentLoadStatus::Status::PENDING_OFFLOAD) {
                        if (status.refcnt_ > 0) {
                            // cancel offload
                            status.status_ = MemorySegmentLoadStatus::Status::LOADED;
                            fprintf(stderr, "Cancel offload %s (%s -> %s)\n", key.to_string().c_str(), 
                                MemorySegmentLoadStatus::StatusString[static_cast<size_t>(current_status)],
                                MemorySegmentLoadStatus::StatusString[static_cast<size_t>(status.status_)]);
                            process_ack(key);
                            CHECK_ZERO(pthread_mutex_unlock(&status.mutex_));
                            continue;
                        }
                        status.status_ = MemorySegmentLoadStatus::Status::BUSY;
                        keys_to_process.push_back(key);
                        processed = true;
                    }
                } else {
                    if (current_status == MemorySegmentLoadStatus::Status::PENDING_LOAD) {
                        status.status_ = MemorySegmentLoadStatus::Status::BUSY;
                        keys_to_process.push_back(key);
                        processed = true;
                    }
                }

                if (!processed) {
                    CHECK_ZERO(pthread_mutex_unlock(&status.mutex_));
                }
            }
        }

        for (auto &key : keys_to_process) {
            // each segment is locked
            auto &status = segment_status(key);
            auto s = status;
            assert(status.status_ == MemorySegmentLoadStatus::Status::BUSY);
            auto prev_status = status.status_;

            if (is_offloading_handler) {
                offload_segment_sync(key);
                status.status_ = MemorySegmentLoadStatus::Status::OFFLOADED;
            } else {
                load_segment_sync(key);
                status.status_ = MemorySegmentLoadStatus::Status::LOADED;
            }

#if PRINT_STATE_TRANSITION
            fprintf(stderr, KCYN "[%d] State Transition %s -> %s : %s\n" KNRM, getpid(), 
                MemorySegmentLoadStatus::StatusString[static_cast<size_t>(prev_status)],
                MemorySegmentLoadStatus::StatusString[static_cast<size_t>(status.status_)], key.to_string().c_str());
#endif

            process_ack(key);
            CHECK_ZERO(pthread_mutex_unlock(&status.mutex_));
        }

        inner_prefetch();

        if (keys_to_process.size())
            cv_offload_.notify_all();

        wait_next = keys_to_process.size() == 0;
    }
}

void GlobalSegmentManager::segment_offloader_main() {
    initialize_thread("GlblLoader");
    segment_loader_offloader_main_impl(true);
}

void GlobalSegmentManager::segment_loader_main() {
    initialize_thread("GlblOffload");
    segment_loader_offloader_main_impl(false);
}

void GlobalSegmentManager::hint_handler_main() {
    initialize_thread("GlblHintHndlr");
    while (!is_finished_) {
        std::unique_lock<std::mutex> ul(mutex_hint_handler_);
        cv_hint_handler_.wait(ul, [this] { return is_finished_ || !queue_hint_handler_.empty(); });

        if (is_finished_)
            break;

        auto key = queue_hint_handler_.front();
        queue_hint_handler_.pop();
        ul.unlock();

        train_progress_estimator_.update(std::get<0>(key), std::get<1>(key), std::get<2>(key));
        if (train_progress_estimator_.is_time_to_expect()) {
            // measure time
            auto start = std::chrono::high_resolution_clock::now();
            auto result = train_progress_estimator_.expect_iter(map_segment_mmaped_info_, num_max_ram_segments_);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            {
                std::unique_lock<std::mutex> ul(mutex_prefetch_hint_);
                map_prefetch_hint_ = result;
            }
            auto end2 = std::chrono::high_resolution_clock::now();
            auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - end);
            fprintf(stderr, KBLU "[%d] Expect Iteration took %ld us, %ld us\n" KNRM, getpid(), duration.count(), duration2.count());
        }
    }
}

void GlobalSegmentManager::offload_segment_sync(MemorySegmentKey key) {
    auto status = segment_status(key);
    assert(status.status_ == MemorySegmentLoadStatus::Status::BUSY);

    std::unique_lock<std::mutex> ul(mutex_segment_mmaped_info_);
    assert(map_segment_mmaped_info_.find(key) != map_segment_mmaped_info_.end());
    auto &mmap_info = map_segment_mmaped_info_[key];
    ul.unlock();

#if PRINT_CPU_SSD_COPY
    fprintf(stderr, KBLU "[%d] CPU->SSD %s\n" KNRM, getpid(), key.to_string().c_str());
#endif

    std::string ssd_path = ssd_path_prefix_ + std::to_string(key.hash());

    // if not gradient
    if (key.type != MemorySegmentKey::Type::GRADIENT) {
        void *dst;

#if (!LAZY_MAP_SSD_TO_REDUCE_NO_MMAP)
        if (mmap_info.ssd_ptr_) {
            dst = mmap_info.ssd_ptr_;
        } else {
#endif
            int ssd_fd = open(ssd_path.c_str(), O_RDWR | O_CREAT, 0644);
            if (ftruncate(ssd_fd, mmap_info.size_) == -1) {
                perror("ftruncate");
                abort();
            }
            dst = mmap(nullptr, mmap_info.size_, PROT_READ | PROT_WRITE, MAP_SHARED, ssd_fd, 0);
            if (dst == (void *) -1) {
                perror("mmap");
                abort();
            } else if (dst == nullptr) {
                perror("mmap");
                abort();
            }
            close(ssd_fd);
#if (!LAZY_MAP_SSD_TO_REDUCE_NO_MMAP)
            mmap_info.ssd_ptr_ = reinterpret_cast<uint8_t *>(dst);
        }
#endif
        
        /* copy data to ssd */
        memcpy(dst, mmap_info.ptr_, mmap_info.size_);
        msync(dst, mmap_info.size_, MS_SYNC);


#if LAZY_MAP_SSD_TO_REDUCE_NO_MMAP
        int ret = munmap(dst, mmap_info.size_);
        if (ret == -1) {
            perror("munmap");
            abort();
        }
#endif

    }

    
    /* shrink shm */
    if (ftruncate(mmap_info.fd_, 0) == -1) {
        perror("ftruncate");
        abort();
    }
}

void GlobalSegmentManager::load_segment_sync(MemorySegmentKey key) {
    auto &status = segment_status(key);
    assert(status.status_ == MemorySegmentLoadStatus::Status::BUSY);

    std::unique_lock<std::mutex> ul(mutex_segment_mmaped_info_);
    assert(map_segment_mmaped_info_.find(key) != map_segment_mmaped_info_.end());
    auto &mmap_info = map_segment_mmaped_info_[key];
    ul.unlock();

    
#if PRINT_CPU_SSD_COPY
        fprintf(stderr, KBLU "[%d] SSD->CPU %s\n" KNRM, getpid(), key.to_string().c_str());
#endif

    if (ftruncate(mmap_info.fd_, mmap_info.size_) == -1) {
        perror("ftruncate");
        abort();
    }

    if (key.type != MemorySegmentKey::Type::GRADIENT) {
        std::string ssd_path = ssd_path_prefix_ + std::to_string(key.hash());
        int ssd_fd;
        if ((ssd_fd = open(ssd_path.c_str(), O_RDONLY)) < 0) {
            perror("open");
            abort();
        }

        void *src;
#if (!LAZY_MAP_SSD_TO_REDUCE_NO_MMAP)        
        if (mmap_info.ssd_ptr_) {
            src = mmap_info.ssd_ptr_;
        } else {
#endif
            src = mmap(nullptr, mmap_info.size_, PROT_READ, MAP_SHARED, ssd_fd, 0);
            close(ssd_fd);
#if (!LAZY_MAP_SSD_TO_REDUCE_NO_MMAP)        
            mmap_info.ssd_ptr_ = reinterpret_cast<uint8_t *>(src);
        }
#endif
        /* copy from ssd to ram */
        memcpy(mmap_info.ptr_, src, mmap_info.size_);


#if (LAZY_MAP_SSD_TO_REDUCE_NO_MMAP)        
        int ret = munmap(src, mmap_info.size_);
        if (ret == -1) {
            perror("munmap");
            abort();
        }
#endif
    }


}

void GlobalSegmentManager::debug_print_segment_stat() const {
    // print segment_stat
    std::stringstream ss;
    ss << "========= SEGMENTS ===========\n";
    for (auto &key : set_segments_) {
        ss << key.to_string() << " : " << 
            MemorySegmentLoadStatus::StatusString[static_cast<size_t>(segment_status(key).status_)]
            << " refcnt=" << segment_status(key).refcnt_ << "\n";
    }
    ss << "========= REGISTERED SEGMENTS ===========\n";
    for (auto &pair : map_segment_mmaped_info_) {
        ss << pair.first.to_string() << " : " << 
            MemorySegmentLoadStatus::StatusString[static_cast<size_t>(segment_status(pair.first).status_)]
            << " refcnt=" << segment_status(pair.first).refcnt_ << "\n";
    }
    ss << "========= PREFETCH HINT ===========\n";
    for (auto &pair : map_prefetch_hint_) {
        ss << pair.first.to_string() << " : " << pair.second << "\n";
    }
    ss << "=====================================";

    std::cerr << ss.str() << std::endl;
}


void TrainProgressEstimator::new_segment(const MemorySegmentKey key) {
    std::unique_lock<std::mutex> ul(mutex_);
    
    num_layers_ = std::max(num_layers_, static_cast<size_t>(key.layer) + 1);
    num_experts_ = std::max(num_experts_, static_cast<size_t>(key.expert) + 1);
    if (key.type == MemorySegmentKey::Type::PARAMETER) {
        if (num_segment_per_layer_.find(key.layer) == num_segment_per_layer_.end()) {
            num_segment_per_layer_[key.layer] = 0;
        }
        num_segment_per_layer_[key.layer]++;
    }
}

void TrainProgressEstimator::update(layer_id_t layer, expert_id_t expert, bool forward) {
    std::unique_lock<std::mutex> ul(mutex_);
    expect_flag_ = current_layer_ != layer || current_forward_ != forward;
    current_layer_ = layer;
    current_expert_ = expert;
    current_forward_ = forward;
}

std::map<MemorySegmentKey, int> TrainProgressEstimator::expect_iter(const std::map<MemorySegmentKey, MmapSegmentInfo> &map_segments, size_t max) const {
    std::map<MemorySegmentKey, int> ret;
    std::unique_lock<std::mutex> ul(mutex_);
    layer_id_t current_layer = current_layer_;
    bool forward = current_forward_;
    ul.unlock();

    /* skip current layer */
    int priority = 1;
    size_t total_cnt = 0;
    for (size_t i = 0; i < num_layers_ * 2; i++) {
        if (forward) {
            if (num_layers_ == current_layer + 1)
                forward = false;
            else
                current_layer++;
        } else {
            if (current_layer == 0)
                forward = true;
            else
                current_layer--;
        }

        size_t cnt = 0;
        for (const auto &segment: map_segments) {
            if (segment.first.layer == current_layer) {
                if (!forward || segment.first.type == MemorySegmentKey::Type::PARAMETER) {
                    if (ret.find(segment.first) == ret.end()) {
                        ret[segment.first] = priority;
                        cnt++;
                        total_cnt++;
                    }
                }
            }
        }
        if (total_cnt > max) {
            break;
        }
        fprintf(stderr, KYEL "Layer %d : %lu segments (%s)\n" KNRM, current_layer, cnt, forward ? "forward" : "backward");
        priority++;
    }

    fprintf(stderr, KYEL "==========================\n" KNRM);

    return ret;
}
