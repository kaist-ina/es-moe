#include "shm_manager.h"
#include <iostream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#undef NDEBUG

int ShmManager::add_slave(int pid) {
    pthread_mutex_lock(&meta_header()->mutex_);

    /* prevent duplicated pid */
    for (size_t i = 0; i < MAX_NUM_SLAVE; i++) {
        if (meta_header()->slave_pids_[i] == pid) {
            pthread_mutex_unlock(&meta_header()->mutex_);
            return -1;
        }
    }

    for (size_t i = 0; i < MAX_NUM_SLAVE; i++) {
        if (meta_header()->slave_pids_[i] == -1) {
           meta_header()->slave_pids_[i] = pid;
           break;
        }
    }

    int slave_id = meta_header()->num_slave_;
    assert(slave_id < MAX_NUM_SLAVE);
    memset(meta_header()->event_ringbuffer_[slave_id], 0, MSG_RINGBUFFER_SIZE * sizeof(ShmEvent));
    meta_header()->event_ringbuffer_start_[slave_id] = 0;
    meta_header()->event_ringbuffer_end_[slave_id] = 0;
    meta_header()->event_ringbuffer_size_[slave_id] = 0;
    
    int ret;
    pthread_mutexattr_t mutex_attr;
    memset(&mutex_attr, 0, sizeof(pthread_mutexattr_t));
    ret = pthread_mutexattr_setpshared(&mutex_attr, PTHREAD_PROCESS_SHARED);
    assert_p(ret == 0);
    ret = pthread_mutex_init(&meta_header()->event_ringbuffer_mutex_[slave_id], &mutex_attr);
    assert_p(ret == 0);

    pthread_condattr_t cond_attr;
    memset(&cond_attr, 0, sizeof(pthread_condattr_t));
    ret = pthread_condattr_setpshared(&cond_attr, PTHREAD_PROCESS_SHARED);
    assert_p(ret == 0);
    ret = pthread_cond_init(&meta_header()->event_ringbuffer_cond_[slave_id], &cond_attr);

    meta_header()->num_slave_++;
    assert(meta_header()->num_slave_ < MAX_NUM_SLAVE);
    pthread_mutex_unlock(&meta_header()->mutex_);
    return slave_id;
}

void ShmManager::master_report_offload_status(int layer_id, int expert, int order, int type, MemorySegmentStatus status) {
    pthread_mutex_lock(&meta_header()->mutex_status_);

    assert(layer_id < MAX_LAYER);
    assert(expert < MAX_EXPERT);
    assert(order < MAX_ORDER);
    assert(type < MAX_TYPE);
    meta_header()->offload_status_[layer_id][expert][order][type] = status;

    // fprintf(stderr, "master report %d %d %d %d %d\n", layer_id, expert, order, type, status);

    pthread_mutex_unlock(&meta_header()->mutex_status_);
    pthread_cond_broadcast(&meta_header()->cond_status_);
}

MemorySegmentStatus ShmManager::get_offload_status(int layer_id, int expert, int order, int type) {
    pthread_mutex_lock(&meta_header()->mutex_status_);

    assert(layer_id < MAX_LAYER);
    assert(expert < MAX_EXPERT);
    assert(order < MAX_ORDER);
    assert(type < MAX_TYPE);
    auto ret = meta_header()->offload_status_[layer_id][expert][order][type];

    pthread_mutex_unlock(&meta_header()->mutex_status_);
    return ret;
}



ShmEvent ShmManager::recv_slave_event(int slave_id) {
    pthread_mutex_lock(&meta_header()->event_ringbuffer_mutex_[slave_id]);

    while (!is_finished_) {
        if (meta_header()->event_ringbuffer_size_[slave_id] == 0) {
            pthread_cond_wait(&meta_header()->event_ringbuffer_cond_[slave_id], &meta_header()->event_ringbuffer_mutex_[slave_id]);
            continue;
        }
        ShmEvent event = meta_header()->event_ringbuffer_[slave_id][meta_header()->event_ringbuffer_start_[slave_id]];
        meta_header()->event_ringbuffer_start_[slave_id] = (meta_header()->event_ringbuffer_start_[slave_id] + 1) % MSG_RINGBUFFER_SIZE;
        meta_header()->event_ringbuffer_size_[slave_id]--;
        pthread_mutex_unlock(&meta_header()->event_ringbuffer_mutex_[slave_id]);
        return event;
    }

    return ShmEvent(-1, -1, EventType::INVALID);
}

void ShmManager::push_slave_event(int slave_id, ShmEvent event) {
    // if (DISABLE_SSD_OFFLOAD)
    //     return;
    pthread_mutex_lock(&meta_header()->event_ringbuffer_mutex_[slave_id]);
    assert(meta_header()->event_ringbuffer_size_[slave_id] < MSG_RINGBUFFER_SIZE);
    meta_header()->event_ringbuffer_size_[slave_id]++;
    meta_header()->event_ringbuffer_[slave_id][meta_header()->event_ringbuffer_end_[slave_id]] = event;
    meta_header()->event_ringbuffer_end_[slave_id] = (meta_header()->event_ringbuffer_end_[slave_id] + 1) % MSG_RINGBUFFER_SIZE;
    pthread_cond_signal(&meta_header()->event_ringbuffer_cond_[slave_id]);
    pthread_mutex_unlock(&meta_header()->event_ringbuffer_mutex_[slave_id]);
}

void ShmManager::broadcast_slave_event(ShmEvent event) {
    // if (DISABLE_SSD_OFFLOAD)
    //     return;
    size_t num_slave = meta_header()->num_slave_;
    for(size_t i = 0; i < num_slave; i++) {
        push_slave_event(i, event);
    }
}



void ShmManager::wait_until_offload_status(int layer_id, int expert, int order, int type, MemorySegmentStatus status) {
    // if (DISABLE_SSD_OFFLOAD)
    //     return;

    assert(layer_id < MAX_LAYER);
    assert(expert < MAX_EXPERT);
    assert(order < MAX_ORDER);
    assert(type < MAX_TYPE);
    pthread_mutex_lock(&meta_header()->mutex_status_);
    while (!is_finished_) {
        if (meta_header()->offload_status_[layer_id][expert][order][type] == status) {
            pthread_mutex_unlock(&meta_header()->mutex_status_);
            return;
        }
        pthread_cond_wait(&meta_header()->cond_status_, &meta_header()->mutex_status_);
    }
}

void ShmManager::send_message(int layer_id, int rank_from, int expert, int iter, MessageOperationType op_type, MessageEventType event, int data) {
    MessageEntry message = {true, layer_id, rank_from, expert, iter, op_type, event, data};
    pthread_mutex_lock(&meta_header()->mutex_);
    assert(meta_header()->msg_ringbuffer_size_ < MSG_RINGBUFFER_SIZE);
    meta_header()->msg_ringbuffer_size_++;
    meta_header()->msg_ringbuffer_[meta_header()->msg_ringbuffer_end_] = message;
    meta_header()->msg_ringbuffer_end_ = (meta_header()->msg_ringbuffer_end_ + 1) % MSG_RINGBUFFER_SIZE;
    pthread_cond_signal(&meta_header()->cond_);
    pthread_mutex_unlock(&meta_header()->mutex_);
}



MessageEntry ShmManager::recv_message(bool async) {
    

    MessageEntry message;

    if (async) {
        pthread_mutex_lock(&meta_header()->mutex_);
        if (meta_header()->msg_ringbuffer_size_ == 0) {
            pthread_mutex_unlock(&meta_header()->mutex_);
            message.valid = false;
            return message;
        }

        message = meta_header()->msg_ringbuffer_[meta_header()->msg_ringbuffer_start_];
        assert(message.valid);
        meta_header()->msg_ringbuffer_start_ = (meta_header()->msg_ringbuffer_start_ + 1) % MSG_RINGBUFFER_SIZE;
        meta_header()->msg_ringbuffer_size_--;
        pthread_mutex_unlock(&meta_header()->mutex_);
        return message;
    }

    pthread_mutex_lock(&meta_header()->mutex_);
    while (!is_finished_) {
        pthread_cond_wait(&meta_header()->cond_, &meta_header()->mutex_);
        if (meta_header()->msg_ringbuffer_size_ == 0) {
            continue;
        }

        message = meta_header()->msg_ringbuffer_[meta_header()->msg_ringbuffer_start_];
        assert(message.valid);
        meta_header()->msg_ringbuffer_start_ = (meta_header()->msg_ringbuffer_start_ + 1) % MSG_RINGBUFFER_SIZE;
        meta_header()->msg_ringbuffer_size_--;
        pthread_mutex_unlock(&meta_header()->mutex_);
        return message;
    }

    pthread_mutex_unlock(&meta_header()->mutex_);
    message.valid = false;
    return message;
}

static_assert(META_BYTES_ALLOC >= sizeof(MemoryPoolMetaHeader), "Allocated Metadata size is too small.");

void ShmManager::finish() {
    is_finished_ = true;
    pthread_cond_broadcast(&meta_header()->cond_);
    pthread_cond_broadcast(&meta_header()->cond_status_);
    pthread_cond_broadcast(&meta_header()->cond_propagate_);
}

void ShmManager::init_shared_memory_master() {

    if (local_session_id_ == 0)
        local_session_id_ = getpid();


    // allocate shared memory for metadata
    {
        // allocate semaphore
        auto sem_name = (SEM_NAME_PREFIX + std::to_string(local_session_id_));

        // try remove
        sem_unlink(sem_name.c_str());
        shm_meta_semaphore_ = sem_open(sem_name.c_str(), O_CREAT, 0666, 1);
        assert(shm_meta_semaphore_ != SEM_FAILED);

        lock();

        std::string shm_meta_name = SHM_NAME_PREFIX + "meta-" + std::to_string(local_session_id_);
        
        std::cout << "Starting SHM Master... at " << shm_meta_name << std::endl;

        // try remove
        shm_unlink(shm_meta_name.c_str());

        int fd = shm_open(shm_meta_name.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
        if (fd == -1) {
            perror("open");
            throw std::runtime_error("Cannot create shared memory segment");
        }

        if (ftruncate64(fd, META_BYTES_ALLOC) == -1) {
            perror("ftruncate");
            throw std::runtime_error("ftruncate failed");
        }
        
        uint8_t *addr = reinterpret_cast<uint8_t *>(mmap(NULL, META_BYTES_ALLOC, PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0));
        if (addr == MAP_FAILED) {
            perror("mmap");
            throw std::runtime_error("mmap failed");
        }

        shm_meta_ptr_ = addr;
        memset(shm_meta_ptr_, 0, META_BYTES_ALLOC);


        int ret;
        pthread_mutexattr_t mutex_attr;
        memset(&mutex_attr, 0, sizeof(pthread_mutexattr_t));
        ret = pthread_mutexattr_setpshared(&mutex_attr, PTHREAD_PROCESS_SHARED);
        assert_p(ret == 0);
        ret = pthread_mutex_init(&meta_header()->mutex_, &mutex_attr);
        assert_p(ret == 0);
        ret = pthread_mutex_init(&meta_header()->mutex_status_, &mutex_attr);
        assert_p(ret == 0);
        ret = pthread_mutex_init(&meta_header()->mutex_propagate_, &mutex_attr);
        assert_p(ret == 0);

        pthread_condattr_t cond_attr;
        memset(&cond_attr, 0, sizeof(pthread_condattr_t));
        ret = pthread_condattr_setpshared(&cond_attr, PTHREAD_PROCESS_SHARED);
        assert_p(ret == 0);
        ret = pthread_cond_init(&meta_header()->cond_, &cond_attr);
        assert_p(ret == 0);
        ret = pthread_cond_init(&meta_header()->cond_status_, &cond_attr);
        assert_p(ret == 0);
        ret = pthread_cond_init(&meta_header()->cond_propagate_, &cond_attr);
        assert_p(ret == 0);

        for(size_t i = 0; i < MAX_PENDING_DEPENDENCIES; i++) {
            meta_header()->fwd_dependency_status_[i] = -1;
            meta_header()->bwd_dependency_status_[i] = -1;
            meta_header()->opt_dependency_status_[i] = -1;
        }

        for(size_t i = 0; i < MAX_NUM_SLAVE; i++) {
            meta_header()->slave_pids_[i] = -1;
        }
    }

    unlock();

    std::cout << "Starting SHM Master OK" << std::endl;
    
}

void ShmManager::lock() {
    if (sem_wait(shm_meta_semaphore_) != 0) {
        perror("sem_wait");
    }
}

void ShmManager::unlock() {
    if (sem_post(shm_meta_semaphore_) != 0) {
        perror("sem_wait");
    }
}


static void hexDump(const void* mem, size_t length) {
    const unsigned char* p = (const unsigned char*)mem;
    for (size_t i = 0; i < length; ++i) {
        fprintf(stderr, "%02x ", p[i]);
        if ((i + 1) % 64 == 0)
            fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
}

#if 0
static void decDump(const void* mem, size_t length) {
    const unsigned char* p = (const unsigned char*)mem;
    for (size_t i = 0; i < length; ++i) {
        fprintf(stderr, "%05d ", p[i]);
        if ((i + 1) % 64 == 0)
            fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
}
#endif

void ShmManager::mark_dependency(DependencyType dep_type, int iter, int layer_id, int expert) {
    lock();
    
    auto dep_status = 
        dep_type == DependencyType::FORWARD ? meta_header()->fwd_dependency_status_ : 
        (dep_type == DependencyType::BACKWARD ? meta_header()->bwd_dependency_status_ : meta_header()->opt_dependency_status_);

    bool found = false;
    // fprintf(stderr, "marking %s iter=%d layer=%d expert=%d\n", dep_type == DependencyType::FWD_BWD ? "FWD_BWD" : "Optim", iter, layer_id, expert);

    // what happens if allow duplicate?
    // for(size_t i = 0; i < MAX_PENDING_DEPENDENCIES; i++) {
    //     if (dep_status[i] == bind_layer_expert(layer_id, expert)) {
    //         found = true;
    //         fprintf(stderr, "marking %s iter=%d layer=%d expert=%d (duplicate)\n", dep_type == DependencyType::FWD_BWD ? "FWD_BWD" : "Optim", iter, layer_id, expert);
    //         break;
    //     }
    // }   

    if (!found) {
        for(size_t i = 0; i < MAX_PENDING_DEPENDENCIES; i++) {
            if (dep_status[i] == -1) {
                dep_status[i] = bind_layer_expert(layer_id, expert);
                found = true;
                break;
            }
        }   
    }
    if (!found) {
        fprintf(stderr, "unmarking %s iter=%d layer=%d expert=%d failed\n", 
            dep_type == DependencyType::FORWARD ? "forward" : (dep_type == DependencyType::BACKWARD ? "backward": "optim"), 
            iter, layer_id, expert);
        hexDump(dep_status, sizeof(int)*MAX_PENDING_DEPENDENCIES);
    }
    assert(found);
    // hexDump(dep_status, sizeof(int)*MAX_PENDING_DEPENDENCIES);
    unlock();
}


void ShmManager::unmark_dependency(DependencyType dep_type, int iter, int layer_id, int expert) {
    lock();
    
    auto dep_status = 
        dep_type == DependencyType::FORWARD ? meta_header()->fwd_dependency_status_ : 
        (dep_type == DependencyType::BACKWARD ? meta_header()->bwd_dependency_status_ : meta_header()->opt_dependency_status_);

    bool found = false;
    // fprintf(stderr, "unmarking %s iter=%d layer=%d expert=%d\n", dep_type == DependencyType::FWD_BWD ? "FWD_BWD" : "Optim", iter, layer_id, expert);
    for(size_t i = 0; i < MAX_PENDING_DEPENDENCIES; i++) {
        if (dep_status[i] == bind_layer_expert(layer_id, expert)) {
            dep_status[i] = -1;
            found = true;
            break;
        }
    }
    if (!found) {
        fprintf(stderr, "unmarking %s iter=%d layer=%d expert=%d failed\n", 
            dep_type == DependencyType::FORWARD ? "forward" : (dep_type == DependencyType::BACKWARD ? "backward": "optim"), 
            iter, layer_id, expert);
        hexDump(dep_status, sizeof(int)*MAX_PENDING_DEPENDENCIES);
    }
    assert(found);
    unlock();
}

std::set<int> ShmManager::get_dependency_set(DependencyType dep_type) {
    std::set<int> ret;
    lock();
    auto dep_status = 
        dep_type == DependencyType::FORWARD ? meta_header()->fwd_dependency_status_ : 
        (dep_type == DependencyType::BACKWARD ? meta_header()->bwd_dependency_status_ : meta_header()->opt_dependency_status_);
        
    for(size_t i = 0; i < MAX_PENDING_DEPENDENCIES; i++) {
        if (dep_status[i] != -1) {
            ret.insert(dep_status[i]);
        }
    }
    unlock();
    return ret;
}

void ShmManager::init_shared_memory_slave() {

    {
        std::string shm_meta_name = SHM_NAME_PREFIX + "meta-" + std::to_string(local_session_id_);
        auto sem_name = (SEM_NAME_PREFIX + std::to_string(local_session_id_));        
        std::cout << "Starting SHM Slave... at " << shm_meta_name << std::endl;

        while (true) {
            shm_meta_semaphore_ = sem_open(sem_name.c_str(), O_CREAT, 0666, 1);
            if (shm_meta_semaphore_ == SEM_FAILED) {
                usleep(1000*100);
            } else {
                break;
            }
        }

        lock();
        unlock();

        int fd;
        while (true) {
            fd = shm_open(shm_meta_name.c_str(), O_RDWR, S_IRUSR | S_IWUSR);
            if (fd == -1) {
                usleep(1000*100);
            } else {
                break;
            }
        }

        // map shared memory to process address space
        uint8_t *addr = reinterpret_cast<uint8_t *>(mmap(NULL, META_BYTES_ALLOC, PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0));
        if (addr == MAP_FAILED)
        {
            perror("mmap");
            throw std::runtime_error("mmap failed");
        }
        
        shm_meta_ptr_ = addr;
        
    }

    std::cout << "Starting SHM Slave OK" << std::endl;
}


void ShmManager::barrier(size_t size) {
    auto *header = reinterpret_cast<MemoryPoolMetaHeader *>(shm_meta_ptr_);

    sem_wait(shm_meta_semaphore_);
    // now metadata is locked!
    bool master = false;

    if (!header->barrier_init) {
        pthread_barrierattr_t barrier_attr;
        pthread_barrierattr_setpshared(&barrier_attr, PTHREAD_PROCESS_SHARED);
        auto ret = pthread_barrier_init(&header->barrier, &barrier_attr, size);
        assert(ret == 0);
        header->barrier_init = true;
        master = true;
    }

    sem_post(shm_meta_semaphore_);    

    pthread_barrier_wait(&header->barrier);
    std::cout << "Barrier OK" << std::endl;

    sem_wait(shm_meta_semaphore_);
    if (master) {
        header->barrier_init = false;
    }
    sem_post(shm_meta_semaphore_);    
}


void ShmManager::wake_other_ranks() {
    auto *header = reinterpret_cast<MemoryPoolMetaHeader *>(shm_meta_ptr_);
    pthread_mutex_lock(&header->mutex_propagate_);

    pthread_cond_broadcast(&header->cond_propagate_);
    pthread_mutex_unlock(&header->mutex_propagate_);
}

bool ShmManager::sleep_for_wake() {
    auto *header = reinterpret_cast<MemoryPoolMetaHeader *>(shm_meta_ptr_);
    pthread_mutex_lock(&header->mutex_propagate_);
    pthread_cond_wait(&header->cond_propagate_, &header->mutex_propagate_);
    pthread_mutex_unlock(&header->mutex_propagate_);
    return true;
}


ShmManager::~ShmManager() {

    sem_close(shm_meta_semaphore_);

    // free shared memory for meta
    if (shm_meta_ptr_ == nullptr)
        return;

    if (munmap(shm_meta_ptr_, META_BYTES_ALLOC) == -1) {
        perror("munmap");
    }

    if (is_master_) {
        const auto shm_name = SHM_NAME_PREFIX + "meta-" + std::to_string(local_session_id_);
        if (shm_unlink(shm_name.c_str()) == -1) {
            perror("unlink");
        }
    
        const auto sem_name = SEM_NAME_PREFIX + std::to_string(local_session_id_);
        sem_unlink(sem_name.c_str());
    }

}



