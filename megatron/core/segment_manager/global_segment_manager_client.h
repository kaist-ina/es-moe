#ifndef _GLOBAL_SEGMENT_MANAGER_CLIENT_H_
#define _GLOBAL_SEGMENT_MANAGER_CLIENT_H_

#include "common.h"
#include "shm_allocator.h"
#include "shm_message_queue.h"
#include <deque>
#include <thread>
#include <map>
#include <condition_variable>


class GlobalSegmentManagerClient {

private:
    std::unique_ptr<ShmMessageQueue<ClientMessage>> request_queue_;
    std::unique_ptr<ShmMessageQueue<ServerMessage>> response_queue_;
    std::unique_ptr<ShmAllocator> shm_allocator_;

    bool is_finished_ = false;
    int client_id_ = -1;
    int rank_ = -1;

    void connect_server();

    /* blocking call that waits until reponse */
    ServerMessage wait_server_response(int seq);
    
    std::unique_ptr<std::thread> wait_response_worker_;
    std::condition_variable cv_wait_response_;
    std::mutex mutex_wait_response_;
    std::map<int, ServerMessage> map_wait_response_;
    void wait_response_worker_main();

    void send_message_impl(const ClientMessage msg, bool blocking);
    

public:
    GlobalSegmentManagerClient(int rank);
    ~GlobalSegmentManagerClient();

    /** register segment, blocking call */
    void segment_register(MemorySegmentKey key, size_t size); 
    
    /** communicate with global manager to acquire segment, blocking call */
    void segment_acquire(MemorySegmentKey key); 
    
    /** communicate with global manager to release segment, non-blocking call */
    void segment_release(MemorySegmentKey key);

    void prefetch_hint(layer_id_t layer, expert_id_t expert, bool forward);

    inline int client_id() const { return client_id_; }

    MemorySegmentLoadStatus segment_status(MemorySegmentKey key);
};

#endif
