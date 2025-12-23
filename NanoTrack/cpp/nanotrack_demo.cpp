#include "nanotrack.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct Args {
    std::string config_path;
};

struct DemoConfig {
    std::string image_dir;
    std::string output_dir;
    bool use_merge = false;
    std::string merge_model;
    std::string backbone = "../models/onnx/nanotrack_backbone.onnx";
    std::string search_backbone;
    std::string head = "../models/onnx/nanotrack_head.onnx";
    bool use_cuda = false;
    cv::Rect2f init_roi{0.f, 0.f, 0.f, 0.f};
    int repeat = 1;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) {
            args.config_path = argv[++i];
        } else {
            std::cout << "未知参数: " << a << "\n";
        }
    }
    return args;
}

bool load_config(const std::string& path, DemoConfig& cfg) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        return false;
    }
    fs["image_dir"] >> cfg.image_dir;
    fs["output_dir"] >> cfg.output_dir;

    // 合并模型配置
    int use_merge = 0;
    fs["use_merge"] >> use_merge;
    cfg.use_merge = use_merge != 0;
    fs["merge_model"] >> cfg.merge_model;

    // 分离模型配置
    fs["backbone"] >> cfg.backbone;
    fs["search_backbone"] >> cfg.search_backbone;
    fs["head"] >> cfg.head;

    int use_cuda = 0;
    fs["use_cuda"] >> use_cuda;
    cfg.use_cuda = use_cuda != 0;
    cv::FileNode roi_node = fs["init_roi"];
    if (!roi_node.empty() && roi_node.isSeq() && roi_node.size() == 4) {
        float x = 0.f;
        float y = 0.f;
        float w = 0.f;
        float h = 0.f;
        roi_node[0] >> x;
        roi_node[1] >> y;
        roi_node[2] >> w;
        roi_node[3] >> h;
        cfg.init_roi = cv::Rect2f(x, y, w, h);
    }
    int repeat = 1;
    if (!fs["repeat"].empty()) {
        fs["repeat"] >> repeat;
    }
    cfg.repeat = std::max(1, repeat);
    return true;
}

std::vector<std::string> list_images(const std::string& dir_path) {
    std::vector<std::string> images;
    if (dir_path.empty()) {
        return images;
    }
    std::filesystem::path dir(dir_path);
    if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
        return images;
    }
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" ||
            ext == ".bmp" || ext == ".tif" || ext == ".tiff") {
            images.push_back(entry.path().string());
        }
    }
    std::sort(images.begin(), images.end());
    return images;
}

std::string format_index_name(int idx, const std::string& ext) {
    std::ostringstream oss;
    oss << "frame_" << std::setw(6) << std::setfill('0') << idx << ext;
    return oss.str();
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    if (args.config_path.empty()) {
        std::cerr << "请使用 --config 指定配置文件路径\n";
        return 1;
    }

    DemoConfig cfg;
    if (!load_config(args.config_path, cfg)) {
        std::cerr << "无法读取配置文件: " << args.config_path << "\n";
        return 1;
    }
    std::cout << "配置文件: " << args.config_path << "\n";
    std::cout << "图片目录: " << cfg.image_dir << "\n";
    std::cout << "输出目录: " << cfg.output_dir << "\n";
    std::cout << "初始 ROI: x=" << cfg.init_roi.x << " y=" << cfg.init_roi.y
              << " w=" << cfg.init_roi.width << " h=" << cfg.init_roi.height << "\n";

    auto image_paths = list_images(cfg.image_dir);
    if (image_paths.empty()) {
        std::cerr << "图片目录为空或不可用: " << cfg.image_dir << "\n";
        return 1;
    }
    std::cout << "图片数量: " << image_paths.size() << "\n";

    if (cfg.init_roi.width <= 0.f || cfg.init_roi.height <= 0.f) {
        std::cerr << "初始 ROI 无效，请在配置文件中设置 init_roi\n";
        return 1;
    }

    std::filesystem::path output_dir(cfg.output_dir);
    if (!cfg.output_dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(output_dir, ec);
        if (ec) {
            std::cerr << "无法创建输出目录: " << cfg.output_dir << "\n";
            return 1;
        }
    } else {
        std::cerr << "输出目录为空，请在配置文件中设置 output_dir\n";
        return 1;
    }

    try {
        std::unique_ptr<NanoTrack> tracker;
        if (cfg.use_merge) {
            std::cout << "使用合并模型: " << cfg.merge_model << "\n";
            if (cfg.merge_model.empty()) {
                std::cerr << "错误: use_merge 为 true 但 merge_model 为空\n";
                return 1;
            }
            tracker = std::make_unique<NanoTrack>(cfg.merge_model, cfg.use_cuda);
        } else {
            std::cout << "使用分离模型: backbone=" << cfg.backbone << ", head=" << cfg.head << "\n";
            if (cfg.backbone.empty() || cfg.head.empty()) {
                std::cerr << "错误: backbone 或 head 路径为空\n";
                return 1;
            }
            tracker = std::make_unique<NanoTrack>(cfg.backbone, cfg.head, cfg.search_backbone, cfg.use_cuda);
        }

        cv::Mat first_frame = cv::imread(image_paths.front(), cv::IMREAD_COLOR);
        if (first_frame.empty()) {
            std::cerr << "读取首张图片失败: " << image_paths.front() << "\n";
            return 1;
        }
        tracker->init(cv::Rect(cfg.init_roi), first_frame);

        int total_frames = static_cast<int>(image_paths.size());
        if (image_paths.size() == 1) {
            total_frames = cfg.repeat;
            std::cout << "单张图片模式，重复次数: " << total_frames << "\n";
        }
        std::cout << "开始跟踪，共处理帧数: " << total_frames << "\n";

        for (int frame_idx = 0; frame_idx < total_frames; ++frame_idx) {
            const std::string& path = image_paths[frame_idx % image_paths.size()];
            cv::Mat frame = cv::imread(path, cv::IMREAD_COLOR);
            if (frame.empty()) {
                std::cerr << "读取图片失败: " << path << "\n";
                continue;
            }

            cv::Rect bbox;
            float score = 1.0f;
            if (frame_idx == 0) {
                bbox = cv::Rect(cfg.init_roi);
            } else {
                bbox = tracker->update(frame);
                score = tracker->last_score();
            }
            cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, cv::format("score: %.3f", score),
                        cv::Point(bbox.x, bbox.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);

            std::filesystem::path out_path = output_dir / format_index_name(frame_idx, ".jpg");
            if (!cv::imwrite(out_path.string(), frame)) {
                std::cerr << "保存结果失败: " << out_path.string() << "\n";
            } else {
                std::cout << "保存结果: " << out_path.string() << "\n";
            }
        }
        std::cout << "处理完成\n";
    } catch (const std::exception& e) {
        std::cerr << "运行失败: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
