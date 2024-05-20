#include "global_segment_manager_client.h"
#include <iostream>
#include <unistd.h>

GlobalSegmentManagerClient::GlobalSegmentManagerClient(int rank) : rank_(rank) {
    connect_server();
    wait_response_worker_ = std::make_unique<std::thread>(&GlobalSegmentManagerClient::wait_response_worker_main, this);
}

GlobalSegmentManagerClient::~GlobalSegmentManagerClient() {
    is_finished_ = true;
    shm_allocator_ = nullptr;

    if (wait_response_worker_) {
        if (request_queue_)
            request_queue_->finish();
        if (response_queue_)
            response_queue_->finish();
        fprintf(stderr, "GlobalSegmentManagerClient::~GlobalSegmentManagerClient: Joining wait_response_worker_\n");
        wait_response_worker_->join();
    }

    shm_allocator_ = nullptr;
    
    fprintf(stderr, "GlobalSegmentManagerClient::~GlobalSegmentManagerClient: Terminate\n");
}

ServerMessage GlobalSegmentManagerClient::wait_server_response(int seq) {
    std::unique_lock<std::mutex> ul(mutex_wait_response_);
    cv_wait_response_.wait(ul, [this, seq] { return map_wait_response_.find(seq) != map_wait_response_.end(); });
    assert(map_wait_response_.find(seq) != map_wait_response_.end());
    ServerMessage msg = map_wait_response_[seq];
    map_wait_response_.erase(seq);
    return msg;
}

void GlobalSegmentManagerClient::wait_response_worker_main() {
    initialize_thread("GlblRespWaiter");
    assert(response_queue_);

    while (!is_finished_) {
        ServerMessage msg = response_queue_->pop();
        if (!msg.valid_) {
            std::cerr << "Invalid message received" << std::endl;
            continue;
        }

        // fprintf(stderr, "[%d : %d] Received response type=%s ack=%d\n",  getpid(), client_id_, ServerMessage::TypeString[static_cast<size_t>(msg.type_)], msg.ack_);

        std::unique_lock<std::mutex> ul(mutex_wait_response_);
        assert(map_wait_response_.find(msg.ack_) == map_wait_response_.end());
        map_wait_response_[msg.ack_] = msg;
        cv_wait_response_.notify_all();
    }
}

void GlobalSegmentManagerClient::connect_server() {
    if (!shm_allocator_) {
        fprintf(stderr, "Initializing SHM Client, pgid=%d, pid=%d\n", getpgid(0), getpid());
        shm_allocator_ = std::make_unique<ShmAllocator>(false, getpgid(0));
    }

    SharedMemoryHeader *shm_header = reinterpret_cast<SharedMemoryHeader *>(shm_allocator_->shm_ptr());
    pthread_mutex_lock(&shm_header->mutex_);

    client_id_ = static_cast<int>(shm_header->num_clients_);
    shm_header->num_clients_++;
    request_queue_ = std::make_unique<ShmMessageQueue<ClientMessage>>(&shm_header->request_qp_[client_id_].uplink_);
    response_queue_ = std::make_unique<ShmMessageQueue<ServerMessage>>(&shm_header->request_qp_[client_id_].downlink_);

    pthread_mutex_unlock(&shm_header->mutex_);
}

MemorySegmentLoadStatus GlobalSegmentManagerClient::segment_status(MemorySegmentKey key) {
    assert(shm_allocator_);
    SharedMemoryHeader *shm_header = reinterpret_cast<SharedMemoryHeader *>(shm_allocator_->shm_ptr());
    return shm_header->offload_status_[key.layer][key.expert][key.order][static_cast<size_t>(key.type)];
}

void GlobalSegmentManagerClient::send_message_impl(const ClientMessage msg, bool blocking) {
    if (is_disable_ssd_offload())
        return;

    assert(request_queue_);
    assert(msg.valid_);
    assert(msg.seq_ > 0);
    
    request_queue_->push(msg);

    pthread_cond_signal(&reinterpret_cast<SharedMemoryHeader *>(shm_allocator_->shm_ptr())->cond_);

    if (blocking) {
        wait_server_response(msg.seq_);
    }
}

void GlobalSegmentManagerClient::segment_register(MemorySegmentKey key, size_t size) {
    send_message_impl(ClientMessage(ClientMessage::Type::REGISTER, key, size), true);
}

void GlobalSegmentManagerClient::segment_acquire(MemorySegmentKey key) {
    send_message_impl(ClientMessage(ClientMessage::Type::ACQUIRE, key), true);
}

void GlobalSegmentManagerClient::segment_release(MemorySegmentKey key) {
    send_message_impl(ClientMessage(ClientMessage::Type::RELEASE, key), false);
}

void GlobalSegmentManagerClient::prefetch_hint(layer_id_t layer, expert_id_t expert, bool forward) {
    send_message_impl(ClientMessage(layer, expert, forward), false);
}