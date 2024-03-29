#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <memory>
#include <vector>

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#include "volrend/common.hpp"
#include "volrend/denoiser/denoiser.hpp"
#include "volrend/internal/auto_filesystem.hpp"
#include "volrend/internal/imwrite.hpp"
#include "volrend/internal/opts.hpp"
#include "volrend/n3tree.hpp"

// clang-format off
#include "volrend/cuda/common.cuh"
#include "volrend/cuda/renderer_kernel.hpp"
// clang-format on

using json = nlohmann::json;

namespace
{
    std::string path_basename(const std::string &str)
    {
        for (size_t i = str.size() - 1; ~i; --i)
        {
            const char c = str[i];
            if (c == '/' || c == '\\')
            {
                return str.substr(i);
            }
        }
        return str;
    }

    std::string remove_ext(const std::string &str)
    {
        for (size_t i = str.size() - 1; ~i; --i)
        {
            const char c = str[i];
            if (c == '.')
            {
                return str.substr(0, i);
            }
        }
        return str;
    }

    int read_transform_matrices(const std::string &path,
                                std::vector<glm::mat4x3> &out)
    {
        std::ifstream ifs(path);
        int cnt = 0;
        if (!ifs)
        {
            fprintf(stderr, "ERROR: '%s' does not exist\n", path.c_str());
            std::exit(1);
        }
        while (ifs)
        {
            glm::mat4x3 tmp;
            float garb;
            // Recall GL is column major
            ifs >> tmp[0][0] >> tmp[1][0] >> tmp[2][0] >> tmp[3][0];
            if (!ifs)
                break;
            ifs >> tmp[0][1] >> tmp[1][1] >> tmp[2][1] >> tmp[3][1];
            ifs >> tmp[0][2] >> tmp[1][2] >> tmp[2][2] >> tmp[3][2];
            if (ifs)
            {
                ifs >> garb >> garb >> garb >> garb;
            }
            ++cnt;
            out.push_back(std::move(tmp));
        }
        return cnt;
    }

    void read_intrins(const std::string &path, float &fx, float &fy)
    {
        std::ifstream ifs(path);
        if (!ifs)
        {
            fprintf(stderr, "ERROR: intrin '%s' does not exist\n", path.c_str());
            std::exit(1);
        }
        float _; // garbage
        ifs >> fx >> _ >> _ >> _;
        ifs >> _ >> fy;
    }

    /*  Functions for Loading LLFF Dataset  */

    inline glm::vec3 _normalize(glm::vec3 vec)
    {
        float norm = sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
        return glm::vec3(vec.x / norm, vec.y / norm, vec.z / norm);
    }
    
    inline glm::mat4x4 _expand_4x3_to_4x4(const glm::mat4x3& mat)
    {
        glm::mat4x4 res(glm::vec4(mat[0], 0.0f), glm::vec4(mat[1], 0.0f), glm::vec4(mat[2], 0.0f), glm::vec4(mat[3], 1.0f));
        return res;
    }

    inline void print_mat4x4(const glm::mat4x4& mat)
    {
        std::cout << "[" << mat[0][0] << "\t" << mat[1][0] << "\t" << mat[2][0] << "\t" << mat[3][0] << "]" << std::endl;
        std::cout << "[" << mat[0][1] << "\t" << mat[1][1] << "\t" << mat[2][1] << "\t" << mat[3][1] << "]" << std::endl;
        std::cout << "[" << mat[0][2] << "\t" << mat[1][2] << "\t" << mat[2][2] << "\t" << mat[3][2] << "]" << std::endl;
        std::cout << "[" << mat[0][3] << "\t" << mat[1][3] << "\t" << mat[2][3] << "\t" << mat[3][3] << "]" << std::endl;
        std::cout << "------------------------" << std::endl;
    }

    inline void print_mat4x3(const glm::mat4x4& mat)
    {
        std::cout << "[" << mat[0][0] << "\t" << mat[1][0] << "\t" << mat[2][0] << "\t" << mat[3][0] << "]" << std::endl;
        std::cout << "[" << mat[0][1] << "\t" << mat[1][1] << "\t" << mat[2][1] << "\t" << mat[3][1] << "]" << std::endl;
        std::cout << "[" << mat[0][2] << "\t" << mat[1][2] << "\t" << mat[2][2] << "\t" << mat[3][2] << "]" << std::endl;
        std::cout << "------------------------" << std::endl;
    }

    inline void print_vec3(const glm::vec3& vec)
    {
        std::cout << "[" << vec.x << "\t" << vec.y << "\t" << vec.z << "]" << std::endl;
        std::cout << "------------------------" << std::endl;
    }

    glm::mat4x3 _viewmatrix(glm::vec3 z, glm::vec3 up, glm::vec3 pos)
    {
        z = _normalize(z);
        glm::vec3 x = _normalize(glm::cross(up, z));
        glm::vec3 y = _normalize(glm::cross(z, x));
        glm::mat4x3 view_mat = glm::mat4x3(x, y, z, pos);
        return view_mat;
    }

    glm::mat4x3 _poses_avg(const std::vector<glm::mat4x3>& trans)
    {
        glm::vec3 z_avg = glm::vec3(0.0), up_avg = glm::vec3(0.0), center_avg = glm::vec3(0.0);
        size_t pose_cnt = trans.size();

        for (const auto& transform : trans)
        {
            glm::vec3 z = glm::vec3(transform[2][0], transform[2][1], transform[2][2]);
            glm::vec3 up = glm::vec3(transform[1][0], transform[1][1], transform[1][2]);
            glm::vec3 center = glm::vec3(transform[3][0], transform[3][1], transform[3][2]);

            z_avg = z_avg + z;
            up_avg = up_avg + up;
            center_avg = center_avg + center;
        }

        z_avg = _normalize(z_avg / (float)pose_cnt);
        up_avg = up_avg / (float)pose_cnt;
        center_avg = center_avg / (float)pose_cnt;

        return _viewmatrix(z_avg, up_avg, center_avg);
    }

    void _recenter_poses(std::vector<glm::mat4x3>& trans)
    {
        glm::mat4x3 poses_avg = _poses_avg(trans);
        glm::mat4x4 c2w_transform = _expand_4x3_to_4x4(poses_avg);
        glm::mat4x4 c2w_inverse = glm::inverse(c2w_transform);

        for (auto& pose : trans)
        {
            glm::mat4x4 pose4 = _expand_4x3_to_4x4(pose);
            pose4 = c2w_inverse * pose4;
            pose = glm::mat4x3(pose4);
        }
    }
    
} // namespace

int main(int argc, char *argv[])
{
    using namespace volrend;

    cxxopts::Options cxxoptions(
        "volrend_headless",
        "Headless PlenOctree volume rendering (c) PlenOctree authors 2021");
    internal::add_common_opts(cxxoptions);

    // clang-format off
    cxxoptions.add_options()
        ("o,write_images", "output directory of images; "
         "if empty, DOES NOT save (for timing only)",
                cxxopts::value<std::string>()->default_value(""))
        ("i,intrin", "intrinsics matrix 4x4; if set, overrides the fx/fy",
                cxxopts::value<std::string>()->default_value(""))
        ("r,reverse_yz", "use OpenCV camera space convention instead of NeRF",
                cxxopts::value<bool>())
        ("scale", "scaling to apply to image",
                cxxopts::value<float>()->default_value("1"))
        ("max_imgs", "max images to render, default no limit",
                cxxopts::value<int>()->default_value("0"))
        ("options", "render options",
                cxxopts::value<std::string>()->default_value(""))
        ("dataset", "dataset type",
                cxxopts::value<std::string>()->default_value("blender"))
        ("ts_module", "path to torchscript module",
                cxxopts::value<std::string>()->default_value(""))
        ("write_buffer", "save auxiliary buffers. "
         "Invalid if output directory is not given.",
                cxxopts::value<bool>())
        ;
    // clang-format on

    cxxoptions.allow_unrecognised_options();

    // Pass a list of camera pose *.txt files after npz file
    // each file should have 4x4 c2w pose matrix
    cxxoptions.positional_help("npz_file [c2w_txt_4x4...]");

    cxxopts::ParseResult args = internal::parse_options(cxxoptions, argc, argv);

    const int device_id = args["gpu"].as<int>();
    if (~device_id)
    {
        cuda(SetDevice(device_id));
    }

    int width = args["width"].as<int>(), height = args["height"].as<int>();
    float fx = args["fx"].as<float>();
    if (fx < 0)
        fx = 1111.11f;
    float fy = args["fy"].as<float>();
    if (fy < 0)
        fy = fx;

    assert(args.unmatched().size() == 1);
    auto poses_path = fs::path(args.unmatched()[0]);

    // Load all transform matrices and intrin
    std::vector<glm::mat4x3> trans;
    std::vector<std::string> basenames;
    std::string dataset_type = args["dataset"].as<std::string>();
    if (dataset_type == "blender") {
        json  poses          = json::parse(std::ifstream(poses_path));
        float camera_angle_x = poses["camera_angle_x"];
        fx = fy      = 0.5f * width / tanf(0.5f * camera_angle_x);

        auto &frames = poses["frames"];
        for (int i = 0; i < frames.size(); i++) {
            auto &m = frames[i]["transform_matrix"];
            // transpose
            glm::mat4x3 tmp;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 4; j++) {
                    tmp[j][i] = m[i][j];
                }
            }
            trans.emplace_back(tmp);
            basenames.push_back("r_" + std::to_string(i));
        }
    } else if (dataset_type == "tt") {
        width = 1920;
        height = 1080;

        // Load intrin matrix
        auto intrin_path = poses_path / ".." / "intrinsics.txt";
        read_intrins(intrin_path.string(), fx, fy);

        // Load poses
        for (const auto &entry : fs::directory_iterator(poses_path)) {
            auto path = entry.path();

            int cnt = read_transform_matrices(path.string(), trans);
            std::string fname = remove_ext(path.filename().string());
            if (cnt == 1) {
                basenames.push_back(fname);
            } else {
                for (int i = 0; i < cnt; ++i) {
                    std::string tmp = std::to_string(i);
                    while (tmp.size() < 6)
                        tmp = "0" + tmp;
                    basenames.push_back(fname + "_" + tmp);
                }
            }
        }
    } else if (dataset_type == "llff") {
        // Read Camera Info
        cnpy::NpyArray poses = cnpy::npy_load(poses_path.string());
        size_t pose_cnt = poses.shape[0];
        size_t LLFF_param_cnt = poses.shape[1];

        auto _get_from = [&](int set, int index_in_set) -> float {
            union {
                float*  ptr_f;
                double* ptr_d;
            };

            bool is_float_ptr = (poses.word_size == 4);
            if (is_float_ptr) {
                ptr_f = poses.data<float>();
            } else {
                assert(poses.word_size == 8);
                ptr_d = poses.data<double>();
            }
            int idx = set * LLFF_param_cnt + index_in_set;
            return is_float_ptr ? ptr_f[idx] : float(ptr_d[idx]);
        };

        constexpr int factor = 4;
        width   = _get_from(0, 9) / factor;
        height  = _get_from(0, 4) / factor;
        fx = fy = _get_from(0, 14) / factor;

        // Correct rotation matrix
        glm::mat4x4 cam_trans(0, 1, 0, 0,
                              -1, 0, 0, 0,
                              0, 0, 1, 0,
                              0, 0, 0, 1);
        glm::mat4x3 temp;
        float bds_min = 1e9f;
        for (int pose_index = 0; pose_index != pose_cnt; ++pose_index)
        {
            float near_clip = _get_from(pose_index, 15);
            if (near_clip < bds_min)
                bds_min = near_clip;
        }

        for (int pose_index = 0; pose_index != pose_cnt; ++pose_index)
        {
            constexpr int row_step = 5;
            for (int i = 0; i != 3; ++i)
                for (int j = 0; j != 4; ++j)
                    temp[j][i] = _get_from(pose_index, i * row_step + j);
            temp = temp * cam_trans;

            float scale = 1.0f / (bds_min * 0.75f);

            for (int i = 0; i != 3; ++i)
                temp[3][i] *= scale;

            trans.emplace_back(temp);
        }

        // Read Basenames
        std::string images_dirname = "images";
        if (factor > 1) {
            images_dirname += "_";
            images_dirname += std::to_string(factor);
        }
        auto images_path = poses_path.parent_path() / images_dirname;

        for (const auto &entry : fs::directory_iterator(images_path)) {
            auto        path  = entry.path();
            std::string fname = remove_ext(path.filename().string());
            basenames.emplace_back(fname);
        }
        std::sort(basenames.begin(), basenames.end());
    }

    // Transform convention
    if (dataset_type == "tt" || args["reverse_yz"].as<bool>()) {
        puts("INFO: Use OpenCV camera convention\n");
        // clang-format off
        glm::mat4x4 cam_trans(1, 0, 0, 0,
                              0, -1, 0, 0,
                              0, 0, -1, 0,
                              0, 0, 0, 1);
        // clang-format on
        for (auto &transform : trans)
        {
            transform = transform * cam_trans;
        }
    } else if (dataset_type == "llff") {
        puts("INFO: Use LLFF camera convention\n");
        _recenter_poses(trans);
    } else {
        puts("INFO: Use NeRF camera convention\n");
    }

    if (trans.size() == 0)
    {
        fputs("WARNING: No camera poses specified, quitting\n", stderr);
        return 1;
    }

    // Load tree
    N3Tree tree(args["file"].as<std::string>());
    if (dataset_type == "llff") {
        tree.use_ndc = true;
        tree.ndc_width = width;
        tree.ndc_height = height;
        tree.ndc_focal = fx;
    }

    {
        float scale = args["scale"].as<float>();
        if (scale != 1.f)
        {
            int owidth = width, oheight = height;
            width *= scale;
            height *= scale;
            fx *= (float)width / owidth;
            fy *= (float)height / oheight;
        }
    }

    {
        int max_imgs = args["max_imgs"].as<int>();
        if (max_imgs > 0 && trans.size() > (size_t)max_imgs)
        {
            trans.resize(max_imgs);
            basenames.resize(max_imgs);
        }
    }

    // Create camera
    Camera camera(width, height, fx, fy);

    // Create buffer for image output
    std::string out_dir = args["write_images"].as<std::string>();
    std::vector<float> buf;
    if (out_dir.size())
    {
        fs::create_directories(out_dir);
        buf.resize(RenderContext::CHANNELS * width * height);
    }

    // Create cuda resources
    cudaArray_t array;
    cudaStream_t stream;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    cuda(MallocArray(&array, &channelDesc, width, height));
    cuda(StreamCreateWithFlags(&stream, cudaStreamDefault));
    cudaArray_t depth_arr = nullptr; // Not using depth buffer

    // Prepare render context
    RenderContext ctx;
    ctx.offscreen = true;
    ctx.update(array, nullptr, width, height);

    // Create denoiser
    // std::cout << Denoiser::test() << std::endl;
    std::unique_ptr<Denoiser> denoiser =
        std::make_unique<Denoiser>(args["ts_module"].as<std::string>());

    // Load render options
    RenderOptions options;
    auto options_path = args["options"].as<std::string>();
    if (!options_path.empty()) {
        auto f_options = std::ifstream(options_path);
        json j_options = json::parse(f_options);
        options = j_options;
    } else {
        options = internal::render_options_from_args(args);
    }

    // Warm up
    camera.transform = trans[0];
    camera._update(false);
    for (int i = 0; i < 100; i++) {
        launch_renderer(tree, camera, options, ctx, stream, true);
        if (options.denoise) {
            denoiser->denoise(camera, ctx, stream);
        }
        // update rng
        ctx.rng.advance();
    }

#ifdef TIME_RECORD_ENABLED
    ctx.timer().reset(stream);
#endif

    // Begin render
    for (size_t i = 0; i < trans.size(); ++i)
    {
        camera.transform = trans[i];
        camera._update(false);

#ifdef TIME_RECORD_ENABLED
        ctx.timer().render_start();
#endif
        launch_renderer(tree, camera, options, ctx, stream, true);
#ifdef TIME_RECORD_ENABLED
        ctx.timer().render_stop();
#endif
        if (options.denoise) {
            denoiser->denoise(camera, ctx, stream);
        }
#ifdef TIME_RECORD_ENABLED
        ctx.timer().record(options.denoise);
#endif

        // update rng
        ctx.rng.advance();

        if (!out_dir.size()) {
            continue;
        }

        if (args["write_buffer"].as<bool>()) {
            // write auxiliary buffer
            const size_t SIZE =
                sizeof(float) * RenderContext::CHANNELS * width * height;
            cuda(Memcpy(
                buf.data(), ctx.aux_buffer, SIZE, cudaMemcpyDeviceToHost));

            auto outfile = std::ofstream(
                out_dir + "/buf_" + basenames[i] + ".bin",
                std::ios::out | std::ios::binary);
            outfile.write((char *)buf.data(), SIZE);
            outfile.close();
        } else {
            // write image
            cuda(Memcpy2DFromArray(
                buf.data(),
                sizeof(float4) * width,
                array,
                0,
                0,
                sizeof(float4) * width,
                height,
                cudaMemcpyDeviceToHost));
            auto buf_uint8 = std::vector<uint8_t>(4 * width * height);
            for (int j = 0; j < buf_uint8.size(); j++) {
                buf_uint8[j] = buf[j] * 255;
            }
            std::string fpath = out_dir + "/" + basenames[i] + ".png";
            internal::write_png_file(fpath, buf_uint8.data(), width, height);
        }

    }

#ifdef TIME_RECORD_ENABLED
    ctx.timer().report();
#endif

    ctx.freeResource();
    cuda(FreeArray(array));
    cuda(StreamDestroy(stream));
}
