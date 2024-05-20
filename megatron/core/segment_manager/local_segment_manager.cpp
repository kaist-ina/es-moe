#include "local_segment_manager.h"
#include <iostream>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/shm.h>
#include <sys/stat.h>

LocalSegmentManager::LocalSegmentManager() {
    if (is_disable_ssd_offload()) {
        return;
    }
}

LocalSegmentManager::~LocalSegmentManager() {

    is_finished_ = true;

    /* Stop threads */
    if (segment_release_worker_) {
        cv_segment_release_worker_.notify_all();
        fprintf(stderr, "LocalSegmentManager::~LocalSegmentManager: Waiting for segment_release_worker_ to join\n");
        segment_release_worker_->join();
    }
    
    if (cuda_release_worker_) {
        cv_cuda_release_worker_.notify_all();
        fprintf(stderr, "LocalSegmentManager::~LocalSegmentManager: Waiting for cuda_release_worker_ to join\n");
        cuda_release_worker_->join();
    }

    fprintf(stderr, "LocalSegmentManager::~LocalSegmentManager: Terminate\n");

}

void LocalSegmentManager::initialize(int rank, std::unordered_map<std::string, int> options, bool force) {


//TODO check force
    if (is_disable_ssd_offload()) {
        return;
    }
    
    assert(!is_initialized_);
    is_initialized_ = true;
    rank_ = rank;

    fprintf(stderr, "Initializing Local Segment Manager rank=%d\n", rank);
    if (rank_ == 0 && !force) {
        global_segment_manager_server_ = std::make_unique<GlobalSegmentManager>();
    } else {
        usleep(500000);
    }

    global_segment_manager_ = std::make_unique<GlobalSegmentManagerClient>(rank);

    /* Start threads */
    segment_release_worker_ = std::make_unique<std::thread>(&LocalSegmentManager::segment_release_worker_main, this);
    cuda_release_worker_ = std::make_unique<std::thread>(&LocalSegmentManager::cuda_release_worker_main, this);
}

void LocalSegmentManager::segment_release_worker_main() {
    initialize_thread("LocalSgmtRel");

    while (!is_finished_) {

        /* Wait for a segment to be released */
        std::unique_lock<std::mutex> lock(mutex_segment_release_worker_);
        cv_segment_release_worker_.wait(lock, [this] {
            return !queue_segment_release_.empty() || is_finished_;
        });

        /* Check if the thread should be finished */
        if (is_finished_) {
            break;
        }

        /* Get the segment from queue */
        for (int step = 0; step < 2; step++) {
            bool try_lock = step == 0;
            bool found = false;
            auto queue_it = queue_segment_release_.begin();

            for (; queue_it != queue_segment_release_.end();) {
                
                /* Get the segment from map */
                std::unique_lock<std::mutex> map_segment_lock(mutex_map_segment_);
                auto it = map_segment_.find(*queue_it);
                assert(it != map_segment_.end());
                auto &segment = it->second;
                map_segment_lock.unlock();

                //trylock segment.mutex() if try_lock is true
                if (try_lock) {
                    std::unique_lock<std::mutex> segment_lock(segment.mutex(), std::try_to_lock);
                
                    if (!segment_lock.owns_lock())
                        continue;
                        
                    // fprintf(stderr, "Releasing %s (TryLock) \n", segment.key().to_string().c_str());

                    found = true;

                    /* Unregister the segment if it is the last time to be released */
                    if (segment.local_refcnt() == 1)
                        cuda_unregister_segment(segment);

                    assert(segment.local_refcnt() > 0);
                    segment.local_refcnt()--;
                    // fprintf(stderr, "[%d] %s local_refcnt=%d\n", getpid(), segment.key().to_string().c_str(), segment.local_refcnt());
                    global_segment_manager_->segment_release(segment.key());
                    
                    queue_it = queue_segment_release_.erase(queue_it);
                    continue;

                } else {
                    // fprintf(stderr, "Releasing %s (Lock) \n", segment.key().to_string().c_str());
                    std::unique_lock<std::mutex> segment_lock(segment.mutex());
                    found = true;

                    /* Unregister the segment if it is the last time to be released */
                    if (segment.local_refcnt() == 1)
                        cuda_unregister_segment(segment);

                    assert(segment.local_refcnt() > 0);
                    segment.local_refcnt()--;
                    global_segment_manager_->segment_release(segment.key());
                    
                    queue_it = queue_segment_release_.erase(queue_it);
                    break;
                }

                ++queue_it;
            }

            if (found)
                break;
        }
    }
}

void LocalSegmentManager::cuda_release_worker_main() {
    initialize_thread("CudaSgmtRel");

    while (!is_finished_) {
        std::unique_lock<std::mutex> ul(mutex_cuda_release_worker_);
        if (set_cuda_release_.size() == 0) {
            cv_cuda_release_worker_.wait(ul, [this] { return set_cuda_release_.size() > 0 || is_finished_; });
            continue;
        }

        assert (set_cuda_release_.size() > 0);
        std::vector<MemorySegmentKey> keys_to_release;

        for(auto it = set_cuda_release_.begin(); it != set_cuda_release_.end();) {
            cudaError_t result = cudaEventQuery(it->first);
            if (result == cudaSuccess) {
                C10_CUDA_CHECK(cudaEventDestroy(it->first));
                keys_to_release.push_back(it->second);
                it = set_cuda_release_.erase(it);
                continue;
            }
            if (result != cudaErrorNotReady) {
                std::cerr << "cudaEventQuery failed: " << cudaGetErrorString(result) << std::endl;
                abort();
            }
            ++it;
        }

        ul.unlock();
        for (auto &key : keys_to_release) {
            local_segment_release(key);
        }
    }
}

void LocalSegmentManager::local_segment_acquire(MemorySegmentKey key) {

    /* Get the segment from map */
    std::unique_lock<std::mutex> map_segment_lock(mutex_map_segment_);
    auto it = map_segment_.find(key);
    assert(it != map_segment_.end());
    auto &segment = it->second;
    map_segment_lock.unlock();

    /* Acquire the segment */
    std::unique_lock<std::mutex> segment_lock(segment.mutex());
    segment.local_refcnt()++;
    assert(segment.local_refcnt() > 0);
    global_segment_manager_->segment_acquire(key);
    auto status = global_segment_manager_->segment_status(key).status_;
    assert(status == MemorySegmentLoadStatus::Status::LOADED);

    /* Register the segment if it is the first time to be acquired */
    if (segment.local_refcnt() == 1) {
        cuda_register_segment(segment);
    }
}

void LocalSegmentManager::local_segment_release(MemorySegmentKey key) {

    /* Check segment from map */
    {
        std::unique_lock<std::mutex> map_segment_lock(mutex_map_segment_);
        assert(map_segment_.find(key) != map_segment_.end());
    }

    std::unique_lock<std::mutex> ul(mutex_segment_release_worker_);
    queue_segment_release_.push_back(key);
    cv_segment_release_worker_.notify_one();
}

void LocalSegmentManager::local_segment_release(MemorySegmentKey key, cudaStream_t stream) {
    
    /* record cuda event */
    cudaEvent_t event;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    C10_CUDA_CHECK(cudaEventRecord(event, stream));
    
    /* Check segment from map */
    {
        std::unique_lock<std::mutex> map_segment_lock(mutex_map_segment_);
        assert(map_segment_.find(key) != map_segment_.end());
    }

    std::unique_lock<std::mutex> ul(mutex_cuda_release_worker_);
    set_cuda_release_.insert(std::make_pair(event, key));
    cv_cuda_release_worker_.notify_one();
}

void LocalSegmentManager::cuda_register_segment(MemorySegment &segment) {
    if (!segment.pin() || segment.currently_pinned()) {
        return;
    }

    assert(!segment.currently_pinned());
    assert(segment.ptr() != nullptr);
    assert(segment.size() > 0);

    /* Register the segment */
    cudaError_t err = cudaHostRegister(segment.ptr(), segment.size(), cudaHostRegisterMapped);
    if (err != cudaSuccess) {
        std::cerr << "Failed to register segment " << segment.key().to_string() << " with error " << cudaGetErrorString(err)
                  << std::endl;
        assert(false);
    }
    segment.currently_pinned() = true;
}

void LocalSegmentManager::cuda_unregister_segment(MemorySegment &segment) {
    if (!segment.pin() || !segment.currently_pinned()) {
        return;
    }

    assert(segment.currently_pinned());
    assert(segment.ptr() != nullptr);
    assert(segment.size() > 0);

    /* Unregister the segment */
    cudaError_t err = cudaHostUnregister(segment.ptr());
    if (err != cudaSuccess) {
        std::cerr << "Failed to unregister segment " << segment.key().to_string() << " with error " << cudaGetErrorString(err)
                  << std::endl;
        assert(false);
    }
    segment.currently_pinned() = false;
}

torch::Tensor LocalSegmentManager::shared_pinned_memory(torch::Tensor tensor, int rank, int layer, int expert, int order, int type, bool pinning) {
    if (!is_disable_ssd_offload() && !is_initialized_) {
        // fprintf(stderr, "LocalSegmentManager is not initialized, force initializing with rank=%d\n", rank);
        initialize(rank, {}, true);
    }
    
    if (!is_disable_ssd_offload()) 
        assert(is_initialized_);

    auto tensor_size = tensor.numel() * tensor.element_size();
    auto key = MemorySegmentKey(layer, expert, order, static_cast<MemorySegmentKey::Type>(type));
    auto memory_segment = alloc_pinned(key, tensor_size, pinning);
    auto shared_tensor = torch::from_blob(memory_segment, tensor.sizes(), tensor.dtype());

    // NOTE: Only Params need initialization
    if ((rank == 0) && static_cast<MemorySegmentKey::Type>(type) == MemorySegmentKey::Type::PARAMETER) {
        shared_tensor.copy_(tensor);
    }

    return shared_tensor;
}

uint8_t *LocalSegmentManager::alloc_pinned(MemorySegmentKey key, size_t size, bool pinning) {
    if (!is_disable_ssd_offload()) 
        assert(is_initialized_);

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

    /* Only Params and Grads should be pinned */
    bool currently_pinned = false;
    if (pinning && (key.type == MemorySegmentKey::Type::PARAMETER || key.type == MemorySegmentKey::Type::GRADIENT)){
        currently_pinned = true;
        C10_CUDA_CHECK(cudaHostRegister((void *) memory_segment_ptr, (size_t) size, cudaHostRegisterMapped));
    }

    {
        std::unique_lock<std::mutex> ul(mutex_map_segment_);
        assert(map_segment_key_.find(memory_segment_ptr) == map_segment_key_.end());
        assert(map_segment_.find(key) == map_segment_.end());
        map_segment_key_[memory_segment_ptr] = key;
        map_segment_[key] = MemorySegment(key, size, memory_segment_ptr, pinning, currently_pinned, shm_fd);
        
        // std::cerr << "Registered Segment " <<  key.to_string() << std::endl;
    }

    global_segment_manager_->segment_register(key, size);
    
    return memory_segment_ptr;
}

void LocalSegmentManager::hook_impl(layer_id_t layer, expert_id_t expert, cudaStream_t stream, bool acquire, std::function<bool(const MemorySegment &segment)> filter_cond) {

    if (is_disable_ssd_offload()) {
        return;
    }

    assert(is_initialized_);

    std::unique_lock<std::mutex> ul(mutex_map_segment_);
    std::vector<MemorySegmentKey> keys_to_process;
    for (auto it = map_segment_.begin(); it != map_segment_.end(); ++it) {
        auto &segment = it->second;
        if (segment.layer() == layer && segment.expert() == expert && filter_cond(segment)) {
            keys_to_process.push_back(segment.key());
        }
    }
    ul.unlock();

    for (auto &key : keys_to_process) {
        if (acquire) {
            local_segment_acquire(key);
        } else {
            if (stream) {
                local_segment_release(key, stream);
            } else {
                local_segment_release(key);
            }
        }
    }
}

void LocalSegmentManager::pre_forward_hook(layer_id_t layer, expert_id_t expert) {
#if PRINT_PYTHON_API_CALL
    fprintf(stderr, KYEL "[%d : %d] [  ] Pre Forward Hook layer=%d expert=%d\n" KNRM , getpid(), global_segment_manager_ ? global_segment_manager_->client_id() : 0, layer, expert);
#endif
    hook_impl(layer, expert, nullptr, true, [](const MemorySegment &segment) {
        return segment.type() == MemorySegmentKey::Type::PARAMETER;
    });
    if (rank_== 0)
        global_segment_manager_->prefetch_hint(layer, expert, true);
#if PRINT_PYTHON_API_CALL
    fprintf(stderr, KGRN "[%d : %d] [OK] Pre Forward Hook layer=%d expert=%d\n" KNRM, getpid(), global_segment_manager_ ? global_segment_manager_->client_id() : 0, layer, expert);
#endif
}

void LocalSegmentManager::post_forward_hook(layer_id_t layer, expert_id_t expert, cudaStream_t stream) {
#if PRINT_PYTHON_API_CALL
    fprintf(stderr, KYEL "[%d : %d] [  ] Post Forward Hook layer=%d expert=%d\n" KNRM, getpid(), global_segment_manager_ ? global_segment_manager_->client_id() : 0, layer, expert);
#endif
    hook_impl(layer, expert, stream, false, [](const MemorySegment &segment) {
        return segment.type() == MemorySegmentKey::Type::PARAMETER;
    });
#if PRINT_PYTHON_API_CALL
    fprintf(stderr, KGRN "[%d : %d] [OK] Post Forward Hook layer=%d expert=%d\n" KNRM, getpid(), global_segment_manager_ ? global_segment_manager_->client_id() : 0, layer, expert);
#endif
}

void LocalSegmentManager::pre_backward_hook(layer_id_t layer, expert_id_t expert) {
#if PRINT_PYTHON_API_CALL
    fprintf(stderr, KYEL "[%d : %d] [  ] Pre Forward Hook layer=%d expert=%d\n" KNRM, getpid(), global_segment_manager_ ? global_segment_manager_->client_id() : 0, layer, expert);
#endif
    hook_impl(layer, expert, nullptr, true, [](const MemorySegment &segment) {
        return segment.type() == MemorySegmentKey::Type::PARAMETER || segment.type() == MemorySegmentKey::Type::GRADIENT;
    });
    if (rank_ == 0)
        global_segment_manager_->prefetch_hint(layer, expert, true);
#if PRINT_PYTHON_API_CALL
    fprintf(stderr, KGRN "[%d : %d] [OK] Pre Backward Hook layer=%d expert=%d\n" KNRM, getpid(), global_segment_manager_ ? global_segment_manager_->client_id() : 0, layer, expert);
#endif
}

void LocalSegmentManager::post_backward_hook(layer_id_t layer, expert_id_t expert, cudaStream_t stream) {
#if PRINT_PYTHON_API_CALL
    fprintf(stderr, KYEL "[%d : %d] [  ] Post Backward Hook layer=%d expert=%d\n" KNRM, getpid(), global_segment_manager_ ? global_segment_manager_->client_id() : 0, layer, expert);
#endif
    hook_impl(layer, expert, stream, false, [](const MemorySegment &segment) {
        return segment.type() == MemorySegmentKey::Type::PARAMETER || segment.type() == MemorySegmentKey::Type::GRADIENT;
    });
#if PRINT_PYTHON_API_CALL
    fprintf(stderr, KGRN "[%d : %d] [OK] Post Backward Hook layer=%d expert=%d\n" KNRM, getpid(), global_segment_manager_ ? global_segment_manager_->client_id() : 0, layer, expert);
#endif
}

void LocalSegmentManager::pre_optimize_hook(layer_id_t layer, expert_id_t expert) {
#if PRINT_PYTHON_API_CALL
    fprintf(stderr, KYEL "[%d : %d] [  ] Pre Optimize Hook layer=%d expert=%d\n" KNRM, getpid(), global_segment_manager_ ? global_segment_manager_->client_id() : 0, layer, expert);
#endif
    hook_impl(layer, expert, nullptr, true, [](const MemorySegment &segment) { return true; });
#if PRINT_PYTHON_API_CALL
    fprintf(stderr, KGRN "[%d : %d] [OK] Pre Optimize Hook layer=%d expert=%d\n" KNRM, getpid(), global_segment_manager_ ? global_segment_manager_->client_id() : 0, layer, expert);
#endif
}

void LocalSegmentManager::post_optimize_hook(layer_id_t layer, expert_id_t expert) {
#if PRINT_PYTHON_API_CALL
    fprintf(stderr, KYEL "[%d : %d] [  ] Post Optimize Hook layer=%d expert=%d\n" KNRM, getpid(), global_segment_manager_ ? global_segment_manager_->client_id() : 0, layer, expert);
#endif
    hook_impl(layer, expert, nullptr, false, [](const MemorySegment &segment) { return true; });
#if PRINT_PYTHON_API_CALL
    fprintf(stderr, KGRN "[%d : %d] [OK] Post Optimize Hook layer=%d expert=%d\n" KNRM, getpid(), global_segment_manager_ ? global_segment_manager_->client_id() : 0, layer, expert);
#endif
}