#ifndef _SHM_ALLOCATOR_H_
#define _SHM_ALLOCATOR_H_
#include <cstdio>
#include <cstdint>
#include <string>
#include <semaphore.h>
#include <functional>
#include <unistd.h>

class ShmAllocator {
    
    static const std::string SHM_NAME_PREFIX;
    static const std::string SEM_NAME_PREFIX;
    static const off64_t SHM_BYTES_ALLOC = 64 * 1024 * 1024;

    uint8_t *shm_meta_ptr_;
    sem_t* shm_meta_semaphore_;
    bool is_master_;
    pid_t local_session_id_;
    bool is_finished_;

    /** Initialize SHM (Master). Must be called by only one process */
    void initialize_master(std::function<void(uint8_t *ptr)> post_initialize);
    
    /** Initialize SHM (Client). Must be called after master initialization */
    void initialize_client(std::function<void(uint8_t *ptr)> post_initialize);


public:
    ShmAllocator(bool is_master = true, pid_t master_pid = 0, std::function<void(uint8_t *ptr)> post_initialize = nullptr) : 
        shm_meta_ptr_(nullptr), shm_meta_semaphore_(nullptr), is_master_(false), 
        local_session_id_(0), is_finished_(false) {
        local_session_id_ = master_pid;
        
        if (local_session_id_ == 0)
            local_session_id_ = getpid();

        if (is_master || master_pid == 0) {
            is_master_ = true;
            initialize_master(post_initialize);
        } else {
            is_master_ = false;
            initialize_client(post_initialize);
        }
    }

    void sem_lock();
    void sem_unlock();

    inline uint8_t *shm_ptr() const { return shm_meta_ptr_; }
};

#endif