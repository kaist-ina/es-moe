// #ifndef ENGINE_THREADPOOL_H
// #define ENGINE_THREADPOOL_H

#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <functional>
#include <future>
#include <csignal>
#include <stdexcept>
#include <iostream>

// https://modoocode.com/285

class ThreadPool {
    private:
        unsigned int n_threads_;
        bool finished_;
        std::vector<std::thread> arr_threads_;
        std::queue<std::function<void()>> task_queue_;
        std::mutex	m_;
        std::condition_variable		cv_;

        int worker_thread_main(size_t thread_idx);

    public:
        ThreadPool();
        ~ThreadPool();

        template <class F, class... Args> std::future<typename std::result_of<F(Args...)>::type> 
        enqueue(F&& f, Args&&... args);

        inline unsigned int n_threads() const { return n_threads_; }
};

template <class F, class... Args>
std::future<typename std::result_of<F(Args...)>::type> ThreadPool::enqueue(F&& f, Args&&... args) {
    
    if (finished_) {
        throw std::runtime_error("ThreadPool has been terminated.");
    }

    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    
    std::future<return_type> job_result_future = task->get_future();
    {
        std::lock_guard<std::mutex> lock(m_);
        task_queue_.push([task]() { (*task)(); });
    }
    cv_.notify_one();

    return job_result_future;
}

ThreadPool::~ThreadPool() {
    finished_ = true;
    cv_.notify_all();
    for (auto & thread : arr_threads_) {
        thread.join();
    }
}

ThreadPool::ThreadPool() : n_threads_(0), finished_(false) {

    n_threads_ = std::thread::hardware_concurrency();

    if (n_threads_ <= 1) {
        throw std::runtime_error("This hardware does not support hardware concurrency.");
    }

    n_threads_ = 3;
    
    /* Spawn threads */
    std::cerr << "Spawning " << n_threads_ << " worker threads...\n";

    arr_threads_.reserve(n_threads_);
    for (unsigned int i = 0; i < n_threads_; ++i) {
        arr_threads_.emplace_back(std::thread(&ThreadPool::worker_thread_main, this, i));
    }

}

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

// #endif
