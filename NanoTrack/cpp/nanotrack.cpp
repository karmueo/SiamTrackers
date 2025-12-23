#include "nanotrack.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

NanoTrack::NanoTrack(const std::string& backbone_path,
                     const std::string& head_path,
                     const std::string& search_backbone_path,
                     bool use_cuda)
    : env_(ORT_LOGGING_LEVEL_WARNING, "nanotrack"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    use_merge_ = false;
    init_sessions(backbone_path, head_path, search_backbone_path, use_cuda);

    window_ = build_window(cfg_.score_size);
    points_ = build_points(cfg_.stride, cfg_.score_size);
}

NanoTrack::NanoTrack(const std::string& merge_path, bool use_cuda)
    : env_(ORT_LOGGING_LEVEL_WARNING, "nanotrack"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    use_merge_ = true;
    init_merge_session(merge_path, use_cuda);

    window_ = build_window(cfg_.score_size);
    points_ = build_points(cfg_.stride, cfg_.score_size);
}

void NanoTrack::init_sessions(const std::string& backbone_path,
                              const std::string& head_path,
                              const std::string& search_backbone_path,
                              bool use_cuda) {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
#ifdef USE_CUDA_EP
    if (use_cuda) {
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = 0;
        try {
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "使用 CUDA Execution Provider\n";
        } catch (const std::exception& e) {
            std::cerr << "CUDA 不可用，回退 CPU: " << e.what() << "\n";
        }
    }
#else
    if (use_cuda) {
        std::cerr << "本构建未开启 USE_CUDA_EP，默认使用 CPU\n";
    }
#endif

    backbone_sess_ = std::make_unique<Ort::Session>(env_, backbone_path.c_str(), session_options);
    if (!search_backbone_path.empty()) {
        search_backbone_sess_ = std::make_unique<Ort::Session>(env_, search_backbone_path.c_str(), session_options);
    }
    head_sess_ = std::make_unique<Ort::Session>(env_, head_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    backbone_input_name_ = backbone_sess_->GetInputNameAllocated(0, allocator).get();
    backbone_output_names_.push_back(backbone_sess_->GetOutputNameAllocated(0, allocator).get());
    if (search_backbone_sess_) {
        search_backbone_input_name_ = search_backbone_sess_->GetInputNameAllocated(0, allocator).get();
        search_backbone_output_names_.push_back(search_backbone_sess_->GetOutputNameAllocated(0, allocator).get());
    }
    head_input_z_name_ = head_sess_->GetInputNameAllocated(0, allocator).get();
    head_input_x_name_ = head_sess_->GetInputNameAllocated(1, allocator).get();
    head_output_names_.push_back(head_sess_->GetOutputNameAllocated(0, allocator).get());
    head_output_names_.push_back(head_sess_->GetOutputNameAllocated(1, allocator).get());

    auto tmpl_shape = head_sess_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    auto srch_shape = head_sess_->GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
    head_template_hw_ = {tmpl_shape.size() >= 4 && tmpl_shape[2] > 0 ? static_cast<int>(tmpl_shape[2]) : -1,
                         tmpl_shape.size() >= 4 && tmpl_shape[3] > 0 ? static_cast<int>(tmpl_shape[3]) : -1};
    head_search_hw_ = {srch_shape.size() >= 4 && srch_shape[2] > 0 ? static_cast<int>(srch_shape[2]) : -1,
                       srch_shape.size() >= 4 && srch_shape[3] > 0 ? static_cast<int>(srch_shape[3]) : -1};

    auto backbone_shape = backbone_sess_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    template_input_hw_ = {backbone_shape.size() >= 4 && backbone_shape[2] > 0 ? static_cast<int>(backbone_shape[2]) : -1,
                          backbone_shape.size() >= 4 && backbone_shape[3] > 0 ? static_cast<int>(backbone_shape[3]) : -1};
    if (search_backbone_sess_) {
        auto sb_shape = search_backbone_sess_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        search_input_hw_ = {sb_shape.size() >= 4 && sb_shape[2] > 0 ? static_cast<int>(sb_shape[2]) : -1,
                            sb_shape.size() >= 4 && sb_shape[3] > 0 ? static_cast<int>(sb_shape[3]) : -1};
    } else {
        search_input_hw_ = template_input_hw_;
    }
}

void NanoTrack::init_merge_session(const std::string& merge_path, bool use_cuda) {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
#ifdef USE_CUDA_EP
    if (use_cuda) {
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = 0;
        try {
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "使用 CUDA Execution Provider\n";
        } catch (const std::exception& e) {
            std::cerr << "CUDA 不可用，回退 CPU: " << e.what() << "\n";
        }
    }
#else
    if (use_cuda) {
        std::cerr << "本构建未开启 USE_CUDA_EP，默认使用 CPU\n";
    }
#endif

    merge_sess_ = std::make_unique<Ort::Session>(env_, merge_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    merge_input_z_name_ = merge_sess_->GetInputNameAllocated(0, allocator).get();
    merge_input_x_name_ = merge_sess_->GetInputNameAllocated(1, allocator).get();
    merge_output_names_.push_back(merge_sess_->GetOutputNameAllocated(0, allocator).get());
    merge_output_names_.push_back(merge_sess_->GetOutputNameAllocated(1, allocator).get());

    // 获取输入形状
    auto z_shape = merge_sess_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    auto x_shape = merge_sess_->GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();

    merge_input_z_hw_ = {z_shape.size() >= 4 && z_shape[2] > 0 ? static_cast<int>(z_shape[2]) : -1,
                         z_shape.size() >= 4 && z_shape[3] > 0 ? static_cast<int>(z_shape[3]) : -1};
    merge_input_x_hw_ = {x_shape.size() >= 4 && x_shape[2] > 0 ? static_cast<int>(x_shape[2]) : -1,
                         x_shape.size() >= 4 && x_shape[3] > 0 ? static_cast<int>(x_shape[3]) : -1};

    std::cout << "合并模型输入: template=" << merge_input_z_hw_.first << "x" << merge_input_z_hw_.second
              << ", search=" << merge_input_x_hw_.first << "x" << merge_input_x_hw_.second << "\n";
}

void NanoTrack::init(const cv::Rect& roi, const cv::Mat& image) {
    cv::Rect2f bbox(static_cast<float>(roi.x),
                    static_cast<float>(roi.y),
                    static_cast<float>(roi.width),
                    static_cast<float>(roi.height));
    center_pos_ = cv::Point2f(bbox.x + (bbox.width - 1) * 0.5f,
                              bbox.y + (bbox.height - 1) * 0.5f);
    size_ = cv::Point2f(bbox.width, bbox.height);
    float w_z = size_.x + cfg_.context_amount * (size_.x + size_.y);
    float h_z = size_.y + cfg_.context_amount * (size_.x + size_.y);
    float s_z = std::sqrt(w_z * h_z);
    channel_average_ = cv::mean(image);

    if (use_merge_) {
        // 合并模式：直接保存模板图像，后续在 update 中一起推理
        if (merge_input_z_hw_.first > 0 && merge_input_z_hw_.first != cfg_.exemplar_size) {
            throw std::runtime_error("合并模型模板输入尺寸与模型静态尺寸不符，模型期待 " +
                                     std::to_string(merge_input_z_hw_.first) + ", 请导出对应尺寸的模型。");
        }
        zf_ = get_subwindow(image, center_pos_, cfg_.exemplar_size, static_cast<int>(std::round(s_z)), channel_average_);
        // 保存模板的 shape
        zf_shape_ = subwindow_shape_;
    } else {
        // 分离模式：分别运行 backbone 和 head
        auto z = get_subwindow(image, center_pos_, cfg_.exemplar_size, static_cast<int>(std::round(s_z)), channel_average_);
        if (template_input_hw_.first > 0 && template_input_hw_.first != cfg_.exemplar_size) {
            throw std::runtime_error("模板输入尺寸与模型静态尺寸不符，模型期待 " +
                                     std::to_string(template_input_hw_.first) + ", 请导出对应尺寸的 backbone 或使用动态/dual backbone。");
        }
        zf_ = run_backbone(z, subwindow_shape_, zf_shape_);
        zf_ = align_feature(zf_, zf_shape_, head_template_hw_, zf_shape_);
    }
    last_score_ = 1.0f;
}

cv::Rect NanoTrack::update(const cv::Mat& image) {
    float w_z = size_.x + cfg_.context_amount * (size_.x + size_.y);
    float h_z = size_.y + cfg_.context_amount * (size_.x + size_.y);
    float s_z = std::sqrt(w_z * h_z);
    float scale_z = cfg_.exemplar_size / s_z;
    float s_x = s_z * (static_cast<float>(cfg_.instance_size) / cfg_.exemplar_size);
    auto x = get_subwindow(image, center_pos_, cfg_.instance_size, static_cast<int>(std::round(s_x)), channel_average_);

    std::vector<float> cls;
    std::vector<float> loc;
    std::vector<int64_t> cls_shape;
    std::vector<int64_t> loc_shape;

    if (use_merge_) {
        // 合并模式：一次推理完成 backbone + head
        if (merge_input_x_hw_.first > 0 && merge_input_x_hw_.first != cfg_.instance_size) {
            throw std::runtime_error("合并模型搜索输入尺寸与模型静态尺寸不符，模型期待 " +
                                     std::to_string(merge_input_x_hw_.first) + ", 请导出对应尺寸的模型。");
        }
        auto outputs = run_merge(zf_, zf_shape_, x, subwindow_shape_, cls_shape, loc_shape);
        cls = std::move(outputs.first);
        loc = std::move(outputs.second);
    } else {
        // 分离模式：分别运行 backbone 和 head
        if (search_input_hw_.first > 0 && search_input_hw_.first != cfg_.instance_size) {
            throw std::runtime_error("搜索输入尺寸与模型静态尺寸不符，模型期待 " +
                                     std::to_string(search_input_hw_.first) + "，请提供 search_backbone 或导出动态输入。");
        }
        std::vector<int64_t> xf_shape;
        auto xf = run_search_backbone(x, subwindow_shape_, xf_shape);
        xf = align_feature(xf, xf_shape, head_search_hw_, xf_shape);

        auto head_outputs = run_head(zf_, zf_shape_, xf, xf_shape, cls_shape, loc_shape);
        cls = std::move(head_outputs.first);
        loc = std::move(head_outputs.second);
    }

    auto score = convert_score(cls, cls_shape);
    auto pred_bbox = convert_bbox(loc, loc_shape);

    auto change = [](float r) { return std::max(r, 1.f / r); };
    auto sz = [](float w, float h) {
        float pad = (w + h) * 0.5f;
        return std::sqrt((w + pad) * (h + pad));
    };

    std::vector<float> penalty(score.size());
    for (size_t i = 0; i < score.size(); ++i) {
        float sc = sz(pred_bbox[2 * score.size() + i], pred_bbox[3 * score.size() + i]) /
                   sz(size_.x * scale_z, size_.y * scale_z);
        float rc = (size_.x / size_.y) /
                   (pred_bbox[2 * score.size() + i] / pred_bbox[3 * score.size() + i]);
        penalty[i] = std::exp(-(change(sc) * change(rc) - 1.f) * cfg_.penalty_k);
    }

    std::vector<float> pscore(score.size());
    for (size_t i = 0; i < score.size(); ++i) {
        float s = penalty[i] * score[i];
        pscore[i] = s * (1.f - cfg_.window_influence) + window_[i] * cfg_.window_influence;
    }

    auto best_iter = std::max_element(pscore.begin(), pscore.end());
    size_t best_idx = static_cast<size_t>(std::distance(pscore.begin(), best_iter));

    cv::Point2f bbox;
    bbox.x = pred_bbox[best_idx] / scale_z + center_pos_.x;
    bbox.y = pred_bbox[score.size() + best_idx] / scale_z + center_pos_.y;

    float width = size_.x * (1 - cfg_.lr) + pred_bbox[2 * score.size() + best_idx] / scale_z * cfg_.lr;
    float height = size_.y * (1 - cfg_.lr) + pred_bbox[3 * score.size() + best_idx] / scale_z * cfg_.lr;

    auto clipped = bbox_clip(bbox.x, bbox.y, width, height, image.rows, image.cols);
    center_pos_ = cv::Point2f(clipped[0], clipped[1]);
    size_ = cv::Point2f(clipped[2], clipped[3]);

    cv::Rect2f rect(center_pos_.x - size_.x * 0.5f,
                    center_pos_.y - size_.y * 0.5f,
                    size_.x, size_.y);
    last_score_ = score[best_idx];
    return cv::Rect(rect);
}

float NanoTrack::last_score() const {
    return last_score_;
}

std::vector<float> NanoTrack::run_backbone(const std::vector<float>& input, const std::vector<int64_t>& input_shape,
                                           std::vector<int64_t>& out_shape) {
    auto input_tensor = tensor_from_buffer(input, input_shape);
    const char* input_names[] = {backbone_input_name_.c_str()};
    const char* output_names[] = {backbone_output_names_[0].c_str()};
    auto outputs = backbone_sess_->Run(Ort::RunOptions{nullptr},
                                       input_names, &input_tensor, 1,
                                       output_names, 1);
    return tensor_to_vector(outputs[0], out_shape);
}

std::vector<float> NanoTrack::run_search_backbone(const std::vector<float>& input, const std::vector<int64_t>& input_shape,
                                                  std::vector<int64_t>& out_shape) {
    auto& search_sess = search_backbone_sess_ ? search_backbone_sess_ : backbone_sess_;
    const std::string& search_name = search_backbone_sess_ ? search_backbone_input_name_ : backbone_input_name_;
    auto input_tensor = tensor_from_buffer(input, input_shape);
    const char* input_names[] = {search_name.c_str()};
    const char* output_names[] = {search_backbone_sess_ ? search_backbone_output_names_[0].c_str()
                                                        : backbone_output_names_[0].c_str()};
    auto outputs = search_sess->Run(Ort::RunOptions{nullptr},
                                    input_names, &input_tensor, 1,
                                    output_names, 1);
    return tensor_to_vector(outputs[0], out_shape);
}

std::pair<std::vector<float>, std::vector<float>> NanoTrack::run_head(const std::vector<float>& zf,
                                                                      const std::vector<int64_t>& zf_shape,
                                                                      const std::vector<float>& xf,
                                                                      const std::vector<int64_t>& xf_shape,
                                                                      std::vector<int64_t>& cls_shape,
                                                                      std::vector<int64_t>& loc_shape) {
    const char* head_input_names[] = {head_input_z_name_.c_str(), head_input_x_name_.c_str()};
    const char* head_output_names[] = {head_output_names_[0].c_str(), head_output_names_[1].c_str()};
    std::array<Ort::Value, 2> head_inputs{
        tensor_from_buffer(zf, zf_shape),
        tensor_from_buffer(xf, xf_shape)};
    auto outputs = head_sess_->Run(Ort::RunOptions{nullptr},
                                   head_input_names, head_inputs.data(), head_inputs.size(),
                                   head_output_names, 2);
    auto cls = tensor_to_vector(outputs[0], cls_shape);
    auto loc = tensor_to_vector(outputs[1], loc_shape);
    return {std::move(cls), std::move(loc)};
}

std::pair<std::vector<float>, std::vector<float>> NanoTrack::run_merge(const std::vector<float>& z,
                                                                       const std::vector<int64_t>& z_shape,
                                                                       const std::vector<float>& x,
                                                                       const std::vector<int64_t>& x_shape,
                                                                       std::vector<int64_t>& cls_shape,
                                                                       std::vector<int64_t>& loc_shape) {
    const char* merge_input_names[] = {merge_input_z_name_.c_str(), merge_input_x_name_.c_str()};
    const char* merge_output_names[] = {merge_output_names_[0].c_str(), merge_output_names_[1].c_str()};
    std::array<Ort::Value, 2> merge_inputs{
        tensor_from_buffer(z, z_shape),
        tensor_from_buffer(x, x_shape)};
    auto outputs = merge_sess_->Run(Ort::RunOptions{nullptr},
                                    merge_input_names, merge_inputs.data(), merge_inputs.size(),
                                    merge_output_names, 2);
    auto cls = tensor_to_vector(outputs[0], cls_shape);
    auto loc = tensor_to_vector(outputs[1], loc_shape);
    return {std::move(cls), std::move(loc)};
}

std::vector<float> NanoTrack::build_window(int size) {
    constexpr float PI = 3.14159265358979323846f;
    std::vector<float> hanning(size);
    for (int i = 0; i < size; ++i) {
        hanning[i] = 0.5f - 0.5f * std::cos(2.f * PI * i / (size - 1));
    }
    std::vector<float> window(size * size);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            window[y * size + x] = hanning[y] * hanning[x];
        }
    }
    return window;
}

std::vector<cv::Point2f> NanoTrack::build_points(int stride, int size) {
    std::vector<cv::Point2f> pts;
    pts.reserve(size * size);
    int ori = -(size / 2) * stride;
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            pts.emplace_back(static_cast<float>(ori + stride * x),
                             static_cast<float>(ori + stride * y));
        }
    }
    return pts;
}

std::vector<float> NanoTrack::get_subwindow(const cv::Mat& im, const cv::Point2f& pos,
                                            int model_sz, int original_sz,
                                            const cv::Scalar& avg_chans) {
    float c = (original_sz + 1) * 0.5f;
    int context_xmin = static_cast<int>(std::floor(pos.x - c + 0.5f));
    int context_xmax = context_xmin + original_sz - 1;
    int context_ymin = static_cast<int>(std::floor(pos.y - c + 0.5f));
    int context_ymax = context_ymin + original_sz - 1;

    int left_pad = std::max(0, -context_xmin);
    int top_pad = std::max(0, -context_ymin);
    int right_pad = std::max(0, context_xmax - im.cols + 1);
    int bottom_pad = std::max(0, context_ymax - im.rows + 1);

    cv::Mat te_im;
    if (left_pad || top_pad || right_pad || bottom_pad) {
        te_im = cv::Mat(im.rows + top_pad + bottom_pad,
                        im.cols + left_pad + right_pad,
                        im.type(), avg_chans);
        im.copyTo(te_im(cv::Rect(left_pad, top_pad, im.cols, im.rows)));
    } else {
        te_im = im;
    }

    context_xmin += left_pad;
    context_xmax += left_pad;
    context_ymin += top_pad;
    context_ymax += top_pad;

    cv::Rect roi(context_xmin, context_ymin,
                 context_xmax - context_xmin + 1,
                 context_ymax - context_ymin + 1);
    cv::Mat im_patch = te_im(roi).clone();
    if (model_sz != original_sz) {
        cv::resize(im_patch, im_patch, cv::Size(model_sz, model_sz));
    }

    std::vector<float> data(1 * 3 * model_sz * model_sz);
    for (int cidx = 0; cidx < 3; ++cidx) {
        for (int y = 0; y < model_sz; ++y) {
            const uint8_t* row_ptr = im_patch.ptr<uint8_t>(y);
            for (int x = 0; x < model_sz; ++x) {
                data[cidx * model_sz * model_sz + y * model_sz + x] =
                    static_cast<float>(row_ptr[x * 3 + cidx]);
            }
        }
    }

    subwindow_shape_ = {1, 3, model_sz, model_sz};
    return data;
}

std::vector<float> NanoTrack::align_feature(const std::vector<float>& feat,
                                            const std::vector<int64_t>& shape,
                                            const std::pair<int, int>& target_hw,
                                            std::vector<int64_t>& out_shape) {
    if (target_hw.first <= 0 || target_hw.second <= 0) {
        out_shape = shape;
        return feat;
    }
    int64_t n = shape[0], c = shape[1], h = shape[2], w = shape[3];
    int t_h = target_hw.first;
    int t_w = target_hw.second;
    if (h == t_h && w == t_w) {
        out_shape = shape;
        return feat;
    }
    int h_start = static_cast<int>((h - t_h) / 2);
    int w_start = static_cast<int>((w - t_w) / 2);
    std::vector<float> cropped(static_cast<size_t>(n * c * t_h * t_w));
    for (int64_t nc = 0; nc < n * c; ++nc) {
        int64_t base_in = nc * h * w;
        int64_t base_out = nc * t_h * t_w;
        for (int yy = 0; yy < t_h; ++yy) {
            for (int xx = 0; xx < t_w; ++xx) {
                cropped[base_out + yy * t_w + xx] =
                    feat[base_in + (yy + h_start) * w + (xx + w_start)];
            }
        }
    }
    out_shape = {n, c, t_h, t_w};
    return cropped;
}

std::vector<float> NanoTrack::convert_score(const std::vector<float>& cls,
                                            const std::vector<int64_t>& shape) const {
    int64_t c = shape[1];
    int64_t h = shape[2];
    int64_t w = shape[3];
    int64_t hw = h * w;
    std::vector<float> score(hw, 0.f);
    if (c == 1) {
        for (int64_t i = 0; i < hw; ++i) {
            score[i] = 1.f / (1.f + std::exp(-cls[i]));
        }
    } else {
        for (int64_t y = 0; y < h; ++y) {
            for (int64_t x = 0; x < w; ++x) {
                int64_t idx = y * w + x;
                float s0 = cls[idx];
                float s1 = cls[hw + idx];
                float e0 = std::exp(s0);
                float e1 = std::exp(s1);
                score[idx] = e1 / (e0 + e1 + 1e-6f);
            }
        }
    }
    return score;
}

std::vector<float> NanoTrack::convert_bbox(const std::vector<float>& loc,
                                           const std::vector<int64_t>& shape) const {
    int64_t h = shape[2];
    int64_t w = shape[3];
    int64_t hw = h * w;
    std::vector<float> bbox(4 * hw);
    for (int64_t y = 0; y < h; ++y) {
        for (int64_t x = 0; x < w; ++x) {
            int64_t idx = y * w + x;
            float l = loc[idx];
            float t = loc[hw + idx];
            float r = loc[2 * hw + idx];
            float b = loc[3 * hw + idx];
            float x1 = points_[idx].x - l;
            float y1 = points_[idx].y - t;
            float x2 = points_[idx].x + r;
            float y2 = points_[idx].y + b;
            bbox[idx] = (x1 + x2) * 0.5f;
            bbox[hw + idx] = (y1 + y2) * 0.5f;
            bbox[2 * hw + idx] = x2 - x1;
            bbox[3 * hw + idx] = y2 - y1;
        }
    }
    return bbox;
}

std::array<float, 4> NanoTrack::bbox_clip(float cx, float cy, float width, float height, int rows, int cols) const {
    cx = std::max(0.f, std::min(cx, static_cast<float>(cols)));
    cy = std::max(0.f, std::min(cy, static_cast<float>(rows)));
    width = std::max(10.f, std::min(width, static_cast<float>(cols)));
    height = std::max(10.f, std::min(height, static_cast<float>(rows)));
    return {cx, cy, width, height};
}

Ort::Value NanoTrack::tensor_from_buffer(const std::vector<float>& data, const std::vector<int64_t>& shape) {
    return Ort::Value::CreateTensor<float>(memory_info_, const_cast<float*>(data.data()),
                                           data.size(), shape.data(), shape.size());
}

std::vector<float> NanoTrack::tensor_to_vector(const Ort::Value& value, std::vector<int64_t>& shape_out) {
    auto info = value.GetTensorTypeAndShapeInfo();
    shape_out = info.GetShape();
    size_t total = info.GetElementCount();
    const float* ptr = value.GetTensorData<float>();
    return std::vector<float>(ptr, ptr + total);
}
