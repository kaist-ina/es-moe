#ifndef _SHM_MESSAGE_QUEUE_H_
#define _SHM_MESSAGE_QUEUE_H_

#include <pthread.h>
#include <cassert>
#include <cstring>

static const size_t MSG_RINGBUFFER_SIZE = 1024;

template <typename T>
struct alignas(4) MessageQueueBuffer {
    bool initialized_;

    pthread_mutex_t mutex_;
    pthread_cond_t cond_;

    uint32_t msg_ringbuffer_start_;
    uint32_t msg_ringbuffer_end_;
    uint32_t msg_ringbuffer_size_;
    T msg_ringbuffer_[MSG_RINGBUFFER_SIZE];
};


template <typename T1, typename T2>
struct MessageQueuePair {
    MessageQueueBuffer<T1> uplink_;
    MessageQueueBuffer<T2> downlink_; 
};

template <typename T>
class ShmMessageQueue {
private:
    MessageQueueBuffer<T>* comm_buffer_;
    bool is_finished_;

public:
    ShmMessageQueue(MessageQueueBuffer<T>* comm_buffer): comm_buffer_(comm_buffer), is_finished_(false) {

        /* must be locked before calling constructor */
        assert(comm_buffer_);
        if (comm_buffer_->initialized_) {
            return;
        }

        int ret;

        pthread_mutexattr_t mutex_attr;
        memset(&mutex_attr, 0, sizeof(pthread_mutexattr_t));
        ret = pthread_mutexattr_setpshared(&mutex_attr, PTHREAD_PROCESS_SHARED);
        assert(ret == 0);
        ret = pthread_mutex_init(&comm_buffer_->mutex_, &mutex_attr);
        assert(ret == 0);

        pthread_condattr_t cond_attr;
        memset(&cond_attr, 0, sizeof(pthread_condattr_t));
        ret = pthread_condattr_setpshared(&cond_attr, PTHREAD_PROCESS_SHARED);
        assert(ret == 0);    
        ret = pthread_cond_init(&comm_buffer_->cond_, &cond_attr);
        assert(ret == 0);

        memset(reinterpret_cast<uint8_t *>(comm_buffer_->msg_ringbuffer_), 0, sizeof(T) * MSG_RINGBUFFER_SIZE);
        comm_buffer_->msg_ringbuffer_start_ = 0;
        comm_buffer_->msg_ringbuffer_end_ = 0;
        comm_buffer_->msg_ringbuffer_size_ = 0;
        comm_buffer_->initialized_ = true;
    }

    ~ShmMessageQueue() {
        is_finished_ = true;
        pthread_cond_broadcast(&comm_buffer_->cond_);
    }

    inline bool initialized() const {
        assert(comm_buffer_);
        return comm_buffer_->initialized_;
    }

    void push(T message) {
        pthread_mutex_lock(&comm_buffer_->mutex_);
        assert(comm_buffer_->msg_ringbuffer_size_ < MSG_RINGBUFFER_SIZE);
        comm_buffer_->msg_ringbuffer_size_++;
        comm_buffer_->msg_ringbuffer_[comm_buffer_->msg_ringbuffer_end_] = message;
        comm_buffer_->msg_ringbuffer_end_ = (comm_buffer_->msg_ringbuffer_end_ + 1) % MSG_RINGBUFFER_SIZE;
        pthread_cond_signal(&comm_buffer_->cond_);
        pthread_mutex_unlock(&comm_buffer_->mutex_);
    }

    size_t size() {
        return comm_buffer_->msg_ringbuffer_size_;
    }

    void finish() {
        is_finished_ = true;
        pthread_cond_broadcast(&comm_buffer_->cond_);
    }

    T pop(bool async = false) {
        pthread_mutex_lock(&comm_buffer_->mutex_);
        if (async) {
            if (comm_buffer_->msg_ringbuffer_size_ == 0) {
                pthread_mutex_unlock(&comm_buffer_->mutex_);
                return T();
            }

            T message = comm_buffer_->msg_ringbuffer_[comm_buffer_->msg_ringbuffer_start_];
            comm_buffer_->msg_ringbuffer_start_ = (comm_buffer_->msg_ringbuffer_start_ + 1) % MSG_RINGBUFFER_SIZE;
            comm_buffer_->msg_ringbuffer_size_--;
            pthread_mutex_unlock(&comm_buffer_->mutex_);
            return message;
        }

        while (!is_finished_) {
            if (comm_buffer_->msg_ringbuffer_size_ == 0) {
                pthread_cond_wait(&comm_buffer_->cond_, &comm_buffer_->mutex_);
                continue;
            }

            T message = comm_buffer_->msg_ringbuffer_[comm_buffer_->msg_ringbuffer_start_];
            comm_buffer_->msg_ringbuffer_start_ = (comm_buffer_->msg_ringbuffer_start_ + 1) % MSG_RINGBUFFER_SIZE;
            comm_buffer_->msg_ringbuffer_size_--;
            pthread_mutex_unlock(&comm_buffer_->mutex_);
            return message;
        }

        pthread_mutex_unlock(&comm_buffer_->mutex_);
        return T();
    }
    
};



#endif