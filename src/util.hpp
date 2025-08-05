#ifndef _UTIL_H_
#define _UTIL_H_

#include <unordered_map>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>

#include <jsoncpp/json/json.h>
#include <opencv2/opencv.hpp> //opencv

#include "tflite/interpreter.h"
#include "tflite/kernels/register.h"
#include "tflite/model.h"

/* Data structures for pipelined inference */
// Data container used to pass results between pipeline stages
struct IntermediateTensor {
    int index;                            // Index of the input image (used for tracking)
    std::vector<float> data;             // Flattened data of input/output tensors
    std::vector<int> tensor_boundaries;  // Marks boundaries between multiple output tensors (if any)
};

// Thread-safe queue for passing data between threads
template <typename T>
class InterStageQueue
{
public:
    void push(T item) // Push an item to the queue
    {
        std::unique_lock<std::mutex> lock(mutex);
        queue.push(std::move(item));
        lock.unlock();
        cond_var.notify_one();
    }

    bool pop(T &item) // Pop an item from the queue
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

    void signal_shutdown() // Signal that no more items will be pushed
    {
        std::unique_lock<std::mutex> lock(mutex);
        shutdown = true;
        lock.unlock();
        cond_var.notify_all();
    }

    size_t size() // Get the number of items in the queue
    {
        std::unique_lock<std::mutex> lock(mutex);
        return queue.size();
    }

private:
    std::queue<T> queue; // C++ standard library queue for storing items
    std::mutex mutex; // Mutex for synchronizing access to the queue
    std::condition_variable cond_var; // Used to block 'pop' until an item is available
    std::atomic<bool> shutdown{false}; // Flag to indicate that no more items will be pushed
};

namespace util
{
    // Alias for high-resolution clock and time point
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    // Struct to store start and end times with indices
    struct TimerResult
    {
        TimePoint start;
        TimePoint end;
        int start_index;
        int stop_index;
    };

    // Global timer map to store timing results
    static std::unordered_map<std::string, TimerResult> timer_map;

    // Global index for identifying timer start/stop order
    static int global_index = 0;

    // Start timing for a given label
    void timer_start(const std::string &label);

    // Stop timing for a given label
    void timer_stop(const std::string &label);

    // Print all timer results stored in timer_map
    void print_all_timers();

    //  Compare throughput between inference driver and pipelined inference driver
    void compare_throughput(const std::string &label1, const std::string &label2, int num_images);

    // Compare the ratio of the longest stage in pipelined inference to the E2E latency of a normal inference
    void compare_latency(const std::vector<std::string> &stage_labels,
                            const std::string &e2e_label);

    //*==========================================*/

    // Load class labels from a JSON file
    // Expects format: { "0": ["n01440764", "tench"], ... }
    std::unordered_map<int, std::string> load_class_labels(const std::string &json_path);

    // Print the shape of a given TfLiteTensor
    void print_tensor_shape(const TfLiteTensor *tensor);

    // Print summary of the loaded TFLite model including tensor and node info
    void print_model_summary(tflite::Interpreter *interpreter, bool delegate_applied);

    // Get indices of top-K elements from a float vector
    std::vector<int> get_topK_indices(const std::vector<float> &data, int k);

    // Compute softmax probabilities from logits
    void softmax(const float *logits, std::vector<float> &probs, int size);

    // Preprocess input image to match model input size (normalization, resize, etc.)
    cv::Mat preprocess_image(cv::Mat &image, int target_height, int target_width);

    // Preprocess function specialized for ResNet-style preprocessing
    cv::Mat preprocess_image_resnet(cv::Mat &image, int target_height, int target_width);

    // Print top predictions with labels and probabilities
    void print_top_predictions(const std::vector<float> &probs, int num_classes, 
                                int top_k, bool show_softmax, 
                                const std::unordered_map<int, std::string> &label_map);
} // namespace util

#endif // _UTIL_H_
