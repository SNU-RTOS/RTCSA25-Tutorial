#include "util.hpp"

// Print the shape of a given TfLiteTensor
void util::print_tensor_shape(const TfLiteTensor *tensor)
{
    printf("[");
    for (int i = 0; i < tensor->dims->size; ++i)
    {
        printf("%d", tensor->dims->data[i]);
        if (i < tensor->dims->size - 1)
            printf(", ");
    }
    printf("]");
}

// Print summary of the loaded TFLite model including tensor and node info
void util::print_model_summary(tflite::Interpreter *interpreter, bool delegate_applied)
{
    printf("\n[INFO] Model Summary \n");
    printf("📥 Input tensor count  : %zu\n", interpreter->inputs().size());
    printf("📤 Output tensor count : %zu\n", interpreter->outputs().size());
    printf("📦 Total tensor count  : %ld\n", interpreter->tensors_size());
    printf("🔧 Node (op) count     : %zu\n", interpreter->nodes_size());
    printf("🧩 Delegate applied    : %s\n", delegate_applied ? "Yes ✅" : "No ❌");
}

// Get indices of top-K elements from a float vector
std::vector<int> util::get_topK_indices(const std::vector<float> &data, int k)
{
    std::vector<int> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(
        indices.begin(), indices.begin() + k, indices.end(),
        [&data](int a, int b)
        { return data[a] > data[b]; });
    indices.resize(k);
    return indices;
}

// Load label file from JSON and return index → label map
std::unordered_map<int, std::string> util::load_class_labels(const std::string &json_path)
{
    std::ifstream ifs(json_path, std::ifstream::binary);
    if (!ifs.is_open())
        throw std::runtime_error("Failed to open label file: " + json_path);

    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errs;

    if (!Json::parseFromStream(builder, ifs, &root, &errs))
        throw std::runtime_error("Failed to parse JSON: " + errs);

    std::unordered_map<int, std::string> label_map;

    for (const auto &key : root.getMemberNames())
    {
        int idx = std::stoi(key);
        if (root[key].isArray() && root[key].size() >= 2)
        {
            label_map[idx] = root[key][1].asString(); // label = second element
        }
    }

    return label_map;
}

// Start timing for a given label
void util::timer_start(const std::string &label)
{
    util::timer_map[label] = util::TimerResult{util::Clock::now(), util::TimePoint{}, util::global_index++};
}

// Stop timing for a given label
void util::timer_stop(const std::string &label)
{
    auto it = util::timer_map.find(label);
    if (it != timer_map.end())
    {
        it->second.end = Clock::now();
        it->second.stop_index = global_index++;
    }
    else
    {
        std::cerr << "[WARN] No active timer for label: " << label << std::endl;
    }
}

// Print all timer results stored in timer_map
void util::print_all_timers()
{
    std::vector<std::pair<std::string, util::TimerResult>> ordered(util::timer_map.begin(), util::timer_map.end());
    std::sort(ordered.begin(), ordered.end(),
              [](const auto &a, const auto &b)
              {
                  return a.second.stop_index < b.second.stop_index; // ascend
              });

    std::cout << "\n[INFO] Elapsed time summary" << std::endl;
    for (const auto &[label, record] : ordered)
    {
        if (record.end != util::TimePoint{})
        {
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(record.end - record.start).count();
            std::cout << "- " << label << " took " << ms << " ms" << std::endl;
        }
    }
}

//  Compare throughput between inference driver and pipelined inference driver
void util::compare_throughput(const std::string &label1, const std::string &label2, int num_images)
{
    auto get_latency = [&](const std::string &label) -> double {
            auto it = util::timer_map.find(label);
            if (it == util::timer_map.end() || it->second.end == util::TimePoint{})
                return -1;
            return static_cast<double>(
                std::chrono::duration_cast<std::chrono::milliseconds>(it->second.end - it->second.start).count());
        };

        double latency1 = get_latency(label1);
        double latency2 = get_latency(label2);

        if (latency1 <= 0 || latency2 <= 0)
        {
            std::cerr << "[ERROR] Could not retrieve valid latencies.\n";
            return;
        }

        double throughput1 = num_images / (latency1 / 1000.0);
        double throughput2 = num_images / (latency2 / 1000.0);
        double improvement = ((throughput2 - throughput1) / throughput1) * 100.0;
        double ratio = throughput2 / throughput1;

        std::cout << "\n[INFO] Throughput comparison\n";
        std::cout << "- " << label1 << ": " << throughput1 << " images/sec (" << latency1 << " ms)\n";
        std::cout << "- " << label2 << ": " << throughput2 << " images/sec (" << latency2 << " ms)\n";
        std::cout << "- Improvement: " << improvement << "%" << std::endl;
        std::cout << "- Ration: " << ratio << std::endl;
}

 // Compare the ratio of the longest stage in pipelined inference to the E2E latency of a normal inference
void util::compare_latency(const std::vector<std::string> &stage_labels,
                            const std::string &e2e_label)
{
    // Helper to get duration in ms
    auto get_latency = [&](const std::string &label) -> double {
        auto it = util::timer_map.find(label);
        if (it == util::timer_map.end() || it->second.end == util::TimePoint{})
            return -1;
        return static_cast<double>(
            std::chrono::duration_cast<std::chrono::milliseconds>(it->second.end - it->second.start).count());
    };

    // Find longest stage
    double max_stage_latency = -1;
    std::string max_stage_name;
    for (const auto &label : stage_labels)
    {
        double latency = get_latency(label);
        if (latency > max_stage_latency)
        {
            max_stage_latency = latency;
            max_stage_name = label;
        }
    }

    if (max_stage_latency <= 0)
    {
        std::cerr << "[ERROR] Could not retrieve valid stage latencies.\n";
        return;
    }

    // Get E2E latency
    double e2e_latency = get_latency(e2e_label);
    if (e2e_latency <= 0)
    {
        std::cerr << "[ERROR] Could not retrieve E2E latency.\n";
        return;
    }

    // Compute ratio
    double ratio = e2e_latency / max_stage_latency;

    std::cout << "\n[INFO] Latency comparison\n";
    std::cout << "- Longest stage: " << max_stage_name << " = " << max_stage_latency << " ms\n";
    std::cout << "- E2E latency (" << e2e_label << "): " << e2e_latency << " ms\n";
    std::cout << "- Ratio: " << ratio << "\n";
}

// Preprocess input image to match model input size (normalization, resize, etc.)
cv::Mat util::preprocess_image(cv::Mat &image, int target_height, int target_width)
{
    int h = image.rows, w = image.cols;
    float scale = 256.0f / std::min(h, w);
    int new_h = static_cast<int>(h * scale);
    int new_w = static_cast<int>(w * scale);

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    int x = (new_w - target_width) / 2;
    int y = (new_h - target_height) / 2;
    cv::Rect crop(x, y, target_width, target_height);

    cv::Mat cropped = resized(crop);
    cv::Mat rgb_image;
    cv::cvtColor(cropped, rgb_image, cv::COLOR_BGR2RGB);

    // Normalize to float32
    cv::Mat float_image;
    rgb_image.convertTo(float_image, CV_32FC3, 1.0 / 255.0);

    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std[3] = {0.229f, 0.224f, 0.225f};

    std::vector<cv::Mat> channels(3);
    cv::split(float_image, channels);
    for (int c = 0; c < 3; ++c)
        channels[c] = (channels[c] - mean[c]) / std[c];
    cv::merge(channels, float_image);

    return float_image;
}

// Preprocess function specialized for ResNet-style preprocessing
cv::Mat util::preprocess_image_resnet(cv::Mat &image, int target_height, int target_width)
{
    // Get original image dimensions
    int h = image.rows, w = image.cols;

    // Compute scale factor to resize the shorter side to 256
    float scale = 256.0f / std::min(h, w);
    int new_h = static_cast<int>(h * scale);
    int new_w = static_cast<int>(w * scale);

    // Resize image while maintaining aspect ratio
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // Calculate crop coordinates to center crop to target size
    int x = (new_w - target_width) / 2;
    int y = (new_h - target_height) / 2;
    cv::Rect crop(x, y, target_width, target_height);

    // Perform center crop
    cv::Mat cropped = resized(crop);

    // Convert to float32, preserve BGR order, scale factor = 1.0
    cv::Mat float_image;
    cropped.convertTo(float_image, CV_32FC3, 1.0);

    // Subtract ImageNet Caffe mean values for B, G, R channels
    const float mean[3] = {103.939f, 116.779f, 123.68f};
    std::vector<cv::Mat> channels(3);
    cv::split(float_image, channels);
    for (int c = 0; c < 3; ++c)
        channels[c] = channels[c] - mean[c];
    cv::merge(channels, float_image);

    // Return preprocessed image
    return float_image;
}

// Compute softmax probabilities from logits
void util::softmax(const float *logits, std::vector<float> &probs, int size)
{
    float max_val = *std::max_element(logits, logits + size);
    float sum = 0.0f;
    for (int i = 0; i < size; ++i)
    {
        probs[i] = std::exp(logits[i] - max_val);
        sum += probs[i];
    }
    if (sum > 0.0f)
    {
        for (int i = 0; i < size; ++i)
        {
            probs[i] /= sum;
        }
    }
}

// Print top predictions with labels and probabilities
void util::print_top_predictions(const std::vector<float> &probs,
                                 int num_classes,
                                 int top_k,
                                 bool show_softmax,
                                 const std::unordered_map<int, std::string> &label_map) {
    std::vector<int> indices(num_classes);
    std::iota(indices.begin(), indices.end(), 0);

    // Sort indices based on corresponding probabilities
    std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
                      [&](int a, int b) { return probs[a] > probs[b]; });

    for (int i = 0; i < top_k; ++i) {
        int idx = indices[i];
        std::cout << "  [Top " << i + 1 << "] Class " << idx;
        
        if (label_map.count(idx)) {
            std::cout << " (" << label_map.at(idx) << ")";
        }

        if (show_softmax) std::cout << " : " << probs[idx];
        std::cout << std::endl;
    }
}

//*==========================================*/
