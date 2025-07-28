// thread_safe_queue.hpp
#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>

// Data container used to pass results between pipeline stages ---
struct IntermediateTensor {
    int index;                            // Index of the input image (used for tracking)
    std::vector<float> data;             // Flattened data (input/output tensor contents)
    std::vector<int> tensor_boundaries;  // Marks boundaries between multiple output tensors (if any)
};


// Thread-safe queue for passing data between threads
template <typename T>
class ThreadSafeQueue
{
private:
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cond_var;
    std::atomic<bool> shutdown{false};

public:
    void push(T item)
    {
        std::unique_lock<std::mutex> lock(mutex);
        queue.push(std::move(item));
        lock.unlock();
        cond_var.notify_one();
    }

    bool pop(T &item)
    {
        std::unique_lock<std::mutex> lock(mutex);
        cond_var.wait(lock, [this]
                      { return !queue.empty() || shutdown; });

        if (shutdown && queue.empty())
        {
            return false;
        }

        item = std::move(queue.front());
        queue.pop();
        return true;
    }

    void signal_shutdown()
    {
        std::unique_lock<std::mutex> lock(mutex);
        shutdown = true;
        lock.unlock();
        cond_var.notify_all();
    }

    size_t size()
    {
        std::unique_lock<std::mutex> lock(mutex);
        return queue.size();
    }
};