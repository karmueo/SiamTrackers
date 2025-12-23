#ifndef NANOTRACK_H
#define NANOTRACK_H

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class NanoTrack {
public:
    // 分离模型模式构造函数
    NanoTrack(const std::string& backbone_path,
              const std::string& head_path,
              const std::string& search_backbone_path = "",
              bool use_cuda = false);

    // 合并模型模式构造函数
    explicit NanoTrack(const std::string& merge_path, bool use_cuda = false);

    void init(const cv::Rect& roi, const cv::Mat& image);
    cv::Rect update(const cv::Mat& image);
    float last_score() const;

    // 获取当前使用的模式
    bool use_merge_model() const { return use_merge_; }

private:
    struct TrackerConfig {
        int exemplar_size = 127;
        int instance_size = 255;
        int score_size = 15;
        int stride = 16;
        float context_amount = 0.5f;
        float window_influence = 0.455f;
        float penalty_k = 0.138f;
        float lr = 0.348f;
    };

    std::vector<float> build_window(int size);
    std::vector<cv::Point2f> build_points(int stride, int size);
    void init_sessions(const std::string& backbone_path,
                       const std::string& head_path,
                       const std::string& search_backbone_path,
                       bool use_cuda);
    void init_merge_session(const std::string& merge_path, bool use_cuda);
    std::vector<float> run_backbone(const std::vector<float>& input, const std::vector<int64_t>& input_shape,
                                    std::vector<int64_t>& out_shape);
    std::vector<float> run_search_backbone(const std::vector<float>& input, const std::vector<int64_t>& input_shape,
                                           std::vector<int64_t>& out_shape);
    std::pair<std::vector<float>, std::vector<float>> run_head(const std::vector<float>& zf,
                                                               const std::vector<int64_t>& zf_shape,
                                                               const std::vector<float>& xf,
                                                               const std::vector<int64_t>& xf_shape,
                                                               std::vector<int64_t>& cls_shape,
                                                               std::vector<int64_t>& loc_shape);
    // 合并模型推理：一次前向传播完成 backbone + head
    std::pair<std::vector<float>, std::vector<float>> run_merge(const std::vector<float>& z,
                                                                const std::vector<int64_t>& z_shape,
                                                                const std::vector<float>& x,
                                                                const std::vector<int64_t>& x_shape,
                                                                std::vector<int64_t>& cls_shape,
                                                                std::vector<int64_t>& loc_shape);
    std::vector<float> get_subwindow(const cv::Mat& im, const cv::Point2f& pos,
                                     int model_sz, int original_sz,
                                     const cv::Scalar& avg_chans);
    std::vector<float> align_feature(const std::vector<float>& feat,
                                     const std::vector<int64_t>& shape,
                                     const std::pair<int, int>& target_hw,
                                     std::vector<int64_t>& out_shape);
    std::vector<float> convert_score(const std::vector<float>& cls,
                                     const std::vector<int64_t>& shape) const;
    std::vector<float> convert_bbox(const std::vector<float>& loc,
                                    const std::vector<int64_t>& shape) const;
    std::array<float, 4> bbox_clip(float cx, float cy, float width, float height, int rows, int cols) const;
    Ort::Value tensor_from_buffer(const std::vector<float>& data, const std::vector<int64_t>& shape);
    std::vector<float> tensor_to_vector(const Ort::Value& value, std::vector<int64_t>& shape_out);

private:
    TrackerConfig cfg_;
    bool use_merge_ = false;
    std::pair<int, int> head_template_hw_{-1, -1};
    std::pair<int, int> head_search_hw_{-1, -1};
    std::pair<int, int> template_input_hw_{-1, -1};
    std::pair<int, int> search_input_hw_{-1, -1};
    std::pair<int, int> merge_input_z_hw_{-1, -1};
    std::pair<int, int> merge_input_x_hw_{-1, -1};
    std::vector<float> window_;
    std::vector<cv::Point2f> points_;
    cv::Point2f center_pos_{0.f, 0.f};
    cv::Point2f size_{0.f, 0.f};
    cv::Scalar channel_average_;
    std::vector<float> zf_;
    std::vector<int64_t> zf_shape_;
    std::vector<int64_t> subwindow_shape_;
    float last_score_ = 0.f;

    Ort::Env env_;
    Ort::MemoryInfo memory_info_;
    std::unique_ptr<Ort::Session> backbone_sess_;
    std::unique_ptr<Ort::Session> search_backbone_sess_;
    std::unique_ptr<Ort::Session> head_sess_;
    std::unique_ptr<Ort::Session> merge_sess_;  // 合并模型 session
    std::string backbone_input_name_;
    std::vector<std::string> backbone_output_names_;
    std::string search_backbone_input_name_;
    std::vector<std::string> search_backbone_output_names_;
    std::string head_input_z_name_;
    std::string head_input_x_name_;
    std::vector<std::string> head_output_names_;
    // 合并模型输入输出名称
    std::string merge_input_z_name_;
    std::string merge_input_x_name_;
    std::vector<std::string> merge_output_names_;
};

#endif
