#include "shm_allocator.h"
#include <cassert>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <semaphore.h>
#include <stdexcept>
#include <iostream>
#include <cstring>


const std::string ShmAllocator::SHM_NAME_PREFIX = "/esmoe-shm-";
const std::string ShmAllocator::SEM_NAME_PREFIX = "/esmoe-sem-";
    

void ShmAllocator::sem_lock() {
    if (sem_wait(shm_meta_semaphore_) != 0) {
        perror("sem_wait");
    }
}

void ShmAllocator::sem_unlock() {
    if (sem_post(shm_meta_semaphore_) != 0) {
        perror("sem_wait");
    }
}

void ShmAllocator::initialize_master(std::function <void(uint8_t *ptr)> post_initialize) {
    assert(local_session_id_ > 0);
    
    auto sem_name = (SEM_NAME_PREFIX + std::to_string(local_session_id_));
    fprintf(stderr, "sem_name: %s\n", sem_name.c_str());

    // try remove old semaphore
    sem_unlink(sem_name.c_str());
    shm_meta_semaphore_ = sem_open(sem_name.c_str(), O_CREAT, 0666, 1);
    assert(shm_meta_semaphore_ != SEM_FAILED);

    sem_lock();

    std::string shm_meta_name = SHM_NAME_PREFIX + "meta-" + std::to_string(local_session_id_);
    shm_unlink(shm_meta_name.c_str());
    int fd = shm_open(shm_meta_name.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    if (fd == -1) {
        perror("open");
        throw std::runtime_error("Cannot create shared memory segment");
    }

    if (ftruncate64(fd, SHM_BYTES_ALLOC) == -1) {
        perror("ftruncate");
        throw std::runtime_error("ftruncate failed");
    }
    
    uint8_t *addr = reinterpret_cast<uint8_t *>(mmap(NULL, SHM_BYTES_ALLOC, PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0));
    if (addr == MAP_FAILED) {
        perror("mmap");
        throw std::runtime_error("mmap failed");
    }

    shm_meta_ptr_ = addr;
    memset(shm_meta_ptr_, 0, SHM_BYTES_ALLOC);

    if (post_initialize) {
        post_initialize(shm_meta_ptr_);
    }

    sem_unlock();

    std::cout << "Starting SHM Master OK" << std::endl;
}


void ShmAllocator::initialize_client(std::function <void(uint8_t *ptr)> post_initialize) {

    std::string shm_meta_name = SHM_NAME_PREFIX + "meta-" + std::to_string(local_session_id_);
    auto sem_name = (SEM_NAME_PREFIX + std::to_string(local_session_id_)); 

    while (true) {
        shm_meta_semaphore_ = sem_open(sem_name.c_str(), O_CREAT, 0666, 1);
        if (shm_meta_semaphore_ == SEM_FAILED) {
            usleep(1000*100);
        } else {
            break;
        }
    }

    sem_lock();

    int fd;
    while (true) {
        fd = shm_open(shm_meta_name.c_str(), O_RDWR, S_IRUSR | S_IWUSR);
        if (fd == -1) {
            usleep(1000*100);
        } else {
            break;
        }
    }

    /* map shared memory to process address space */
    uint8_t *addr = reinterpret_cast<uint8_t *>(mmap(NULL, SHM_BYTES_ALLOC, PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0));
    if (addr == MAP_FAILED)
    {
        perror("mmap");
        throw std::runtime_error("mmap failed");
    }
    
    shm_meta_ptr_ = addr;

    if (post_initialize) {
        post_initialize(shm_meta_ptr_);
    }

    sem_unlock();
}
