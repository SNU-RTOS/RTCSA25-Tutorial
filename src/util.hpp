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

#include <jsoncpp/json/json.h>
#include <opencv2/opencv.hpp> //opencv

#include "tflite/interpreter.h"
#include "tflite/kernels/register.h"
#include "tflite/model.h"

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

    // Print execution plan (operator ordering) of the model
    void PrintExecutionPlanOps(std::unique_ptr<tflite::Interpreter>& interpreter);
} // namespace util


#endif // _UTIL_H_
