#include <csignal>
#include <stdexcept>
#include <iostream>
#include "threadpool.h"


int ThreadPool::worker_thread_main(size_t thread_idx) {
    sigset_t mask;
    sigemptyset (&mask);
    sigaddset (&mask, SIGTERM);
    sigaddset (&mask, SIGHUP);
    pthread_sigmask(SIG_BLOCK, &mask, NULL);

    while (true) {
        std::function<void()> task = [] () -> void {};
        {
            std::unique_lock<std::mutex> ul(m_);
            while (task_queue_.empty() && !finished_)
                cv_.wait(ul, [&] { return !task_queue_.empty() || finished_; });
            if (finished_)
                break;
            task = std::move(task_queue_.front()); 
            task_queue_.pop();
        }
        task();
    }

    return 0;
}


ThreadPool::ThreadPool() : n_threads_(0), finished_(false) {

    n_threads_ = std::thread::hardware_concurrency();

    if (n_threads_ <= 1) {
        throw std::runtime_error("This hardware does not support hardware concurrency.");
    }

    n_threads_ = 10;
    
    /* Spawn threads */
    std::cerr << "Spawning " << n_threads_ << " worker threads...\n";

    arr_threads_.reserve(n_threads_);
    for (unsigned int i = 0; i < n_threads_; ++i) {
        arr_threads_.emplace_back(std::thread(&ThreadPool::worker_thread_main, this, i));
    }

}

ThreadPool::~ThreadPool() {
    finished_ = true;
    cv_.notify_all();
    for (auto & thread : arr_threads_) {
        thread.join();
    }
}

