#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <stdint.h>
#include <iostream>

#include "volrend/internal/auto_filesystem.hpp"

#include "volrend/common.hpp"
#include "volrend/n3tree.hpp"

#include "volrend/internal/opts.hpp"

#include "volrend/cuda/common.cuh"
#include "volrend/cuda/renderer_kernel.hpp"
#include "volrend/internal/imwrite.hpp"
#include "json.hpp"

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
    // {
    //     // Load intrin matrix
    //     std::string intrin_path = args["intrin"].as<std::string>();
    //     if (intrin_path.size())
    //     {
    //         read_intrins(intrin_path, fx, fy);
    //     }
    // }

    // Load all transform matrices and intrin
    std::vector<glm::mat4x3> trans;
    std::vector<std::string> basenames;
    // for (auto path : args.unmatched())
    // {
    //     int cnt = read_transform_matrices(path, trans);
    //     std::string fname = remove_ext(path_basename(path));
    //     if (cnt == 1)
    //     {
    //         basenames.push_back(fname);
    //     }
    //     else
    //     {
    //         for (int i = 0; i < cnt; ++i)
    //         {
    //             std::string tmp = std::to_string(i);
    //             while (tmp.size() < 6)
    //                 tmp = "0" + tmp;
    //             basenames.push_back(fname + "_" + tmp);
    //         }
    //     }
    // }
    assert(args.unmatched().size() == 1);
    json poses = json::parse(std::ifstream(args.unmatched()[0]));
    float camera_angle_x = poses["camera_angle_x"];
    fx = fy = 0.5f * width / tanf(0.5f * camera_angle_x);
    auto& frames = poses["frames"];
    for (int i = 0; i < frames.size(); i++) {
        auto &m = frames[i]["transform_matrix"];
        glm::mat4x3 tmp;
        // transpose
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                tmp[j][i] = m[i][j];
            }
        }
        trans.emplace_back(tmp);
        basenames.push_back("r_" + std::to_string(i));
    }

    if (args["reverse_yz"].as<bool>())
    {
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
    }
    else
    {
        puts("INFO: Use NeRF camera convention\n");
    }

    if (trans.size() == 0)
    {
        fputs("WARNING: No camera poses specified, quitting\n", stderr);
        return 1;
    }
    std::string out_dir = args["write_images"].as<std::string>();

    N3Tree tree(args["file"].as<std::string>());


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

    Camera camera(width, height, fx, fy);
    cudaArray_t array;
    cudaStream_t stream;

    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc<float4>();

    std::vector<float> buf;
    if (out_dir.size())
    {
        std::filesystem::create_directories(out_dir);
        // std::filesystem::create_directories(out_dir + "/plenoctree");
        // std::filesystem::create_directories(out_dir + "/copy");
        // std::filesystem::create_directories(out_dir + "/trans");
        // std::filesystem::create_directories(out_dir + "/rotate");
        buf.resize(4 * width * height);
    }

    cuda(MallocArray(&array, &channelDesc, width, height));
    cuda(StreamCreateWithFlags(&stream, cudaStreamDefault));
    cudaArray_t depth_arr = nullptr; // Not using depth buffer

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    RenderOptions options = internal::render_options_from_args(args);
    RenderContext ctx;
    ctx.resize(width, height);

    // // warm up
    // ctx.clearHistory();
    // camera.transform = trans[0];
    // camera._update(false);
    // for (size_t i = 0; i < 1024; ++i)
    // {
    //     launch_renderer(tree, camera, options, array, depth_arr, stream, ctx, true);
    // }

    // options.denoise = false;
    // options.clamp_support = 2;
    // options.clamp_k = 1.0f;
    // options.depth_diff_thresh = 100000.0f;
    // options.prev_weight = 0.f;

    // // render poses
    // int interpol_num = 2;
    // int warm_up_num = 10;
    // int copy_num = 1;
    // float max_trans_legth = 50;
    // float max_rotate_degree = 5;
    // max_rotate_degree = max_rotate_degree * 3.14159 / 180;
    // float delta_degree = max_rotate_degree / (float)interpol_num;
    // float delta_len = max_trans_legth / float(interpol_num);
    // cudaEventRecord(start);
    // for (size_t i = 0; i < trans.size(); ++i)
    // {
    //     for (int trans_i = 0; trans_i < 3; trans_i++)
    //     {
    //         if (trans_i == 0)
    //         {
    //             ctx.clearHistory();
    //             camera.transform = trans[i];
    //             camera._update(false);
    //             for (int inter_i = 0; inter_i < copy_num; inter_i++)
    //             {
    //                 launch_renderer(tree, camera, options, array, depth_arr, stream, ctx, true);
    //             }
    //             if (out_dir.size())
    //             {
    //                 cuda(Memcpy2DFromArrayAsync(buf.data(), sizeof(float4) * width, array, 0, 0,
    //                                             sizeof(float4) * width, height,
    //                                             cudaMemcpyDeviceToHost, stream));
    //                 auto buf_uint8 = std::vector<uint8_t>(4 * width * height);
    //                 for (int j = 0; j < buf.size(); j++)
    //                 {
    //                     buf_uint8[j] = buf[j] * 255;
    //                 }
    //                 std::string fpath = out_dir + "/copy/" + basenames[i] + ".png";
    //                 internal::write_png_file(fpath, buf_uint8.data(), width, height);
    //             }
    //         }
    //         else if (trans_i == 1)
    //         {
    //             trans[i][3][0] -= interpol_num * delta_len;
    //             trans[i][3][0] += delta_len;
    //             ctx.clearHistory();
    //             camera.transform = trans[i];
    //             camera._update(false);
    //             for (size_t i = 0; i < warm_up_num; ++i)
    //             {
    //                 launch_renderer(tree, camera, options, array, depth_arr, stream, ctx, true);
    //             }
    //             // if (out_dir.size())
    //             // {
    //             //     cuda(Memcpy2DFromArrayAsync(buf.data(), sizeof(float4) * width, array, 0, 0,
    //             //                                 sizeof(float4) * width, height,
    //             //                                 cudaMemcpyDeviceToHost, stream));
    //             //     auto buf_uint8 = std::vector<uint8_t>(4 * width * height);
    //             //     for (int j = 0; j < buf.size(); j++)
    //             //     {
    //             //         buf_uint8[j] = buf[j] * 255;
    //             //     }
    //             //     std::string fpath = out_dir + "/trans/" + basenames[i] + ".png";
    //             //     internal::write_png_file(fpath, buf_uint8.data(), width, height);
    //             // }
    //             for (int inter_i = 1; inter_i < interpol_num; inter_i++)
    //             {
    //                 trans[i][3][0] += delta_len;
    //                 camera.transform = trans[i];
    //                 camera._update(false);
    //                 launch_renderer(tree, camera, options, array, depth_arr, stream, ctx, true);
    //             }
    //             if (out_dir.size())
    //             {
    //                 cuda(Memcpy2DFromArrayAsync(buf.data(), sizeof(float4) * width, array, 0, 0,
    //                                             sizeof(float4) * width, height,
    //                                             cudaMemcpyDeviceToHost, stream));
    //                 auto buf_uint8 = std::vector<uint8_t>(4 * width * height);
    //                 for (int j = 0; j < buf.size(); j++)
    //                 {
    //                     buf_uint8[j] = buf[j] * 255;
    //                 }
    //                 std::string fpath = out_dir + "/trans/" + basenames[i] + ".png";
    //                 internal::write_png_file(fpath, buf_uint8.data(), width, height);
    //             }
    //         }
    //         else
    //         {
    //             std::vector<glm::mat4x3> trans_series;
    //             trans_series.resize(interpol_num);
    //             for (int inter_i = 0; inter_i < interpol_num; inter_i++)
    //             {
    //                 for (int tmp = 0; tmp < 3; tmp++)
    //                 {
    //                     trans_series[inter_i][0][tmp] = trans[i][0][tmp] * cos(delta_degree * inter_i) + trans[i][1][tmp] * sin(delta_degree * inter_i);
    //                     trans_series[inter_i][1][tmp] = -trans[i][0][tmp] * sin(delta_degree * inter_i) + trans[i][1][tmp] * cos(delta_degree * inter_i);
    //                     trans_series[inter_i][2][tmp] = trans[i][2][tmp];
    //                     trans_series[inter_i][3][tmp] = trans[i][3][tmp];
    //                 }
    //             }
    //             camera.transform = trans_series[interpol_num - 1];
    //             camera._update(false);
    //             ctx.clearHistory();
    //             for (size_t i = 0; i < warm_up_num; ++i)
    //             {
    //                 launch_renderer(tree, camera, options, array, depth_arr, stream, ctx, true);
    //             }
    //             for (int inter_i = interpol_num - 2; inter_i >= 0; inter_i--)
    //             {
    //                 camera.transform = trans_series[inter_i];
    //                 camera._update(false);
    //                 launch_renderer(tree, camera, options, array, depth_arr, stream, ctx, true);
    //             }
    //             if (out_dir.size())
    //             {
    //                 cuda(Memcpy2DFromArrayAsync(buf.data(), sizeof(float4) * width, array, 0, 0,
    //                                             sizeof(float4) * width, height,
    //                                             cudaMemcpyDeviceToHost, stream));
    //                 auto buf_uint8 = std::vector<uint8_t>(4 * width * height);
    //                 for (int j = 0; j < buf.size(); j++)
    //                 {
    //                     buf_uint8[j] = buf[j] * 255;
    //                 }
    //                 std::string fpath = out_dir + "/rotate/" + basenames[i] + ".png";
    //                 internal::write_png_file(fpath, buf_uint8.data(), width, height);
    //             }
    //         }
    //     }
    // }
    
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // float milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // size_t cnt = trans.size() * (copy_num + (interpol_num - 1 + warm_up_num) * 2);
    // milliseconds = milliseconds / cnt;

    // printf("%.10f ms per frame\n", milliseconds);
    // printf("%.10f fps\n", 1000.f / milliseconds);

    options.delta_tracking = true;
    options.denoise = false;
    ctx.clearHistory();
    for (size_t i = 0; i < trans.size(); ++i)
    {
        camera.transform = trans[i];
        camera._update(false);

        launch_renderer(tree, camera, options, array, depth_arr, stream, ctx, true);
        if (out_dir.size())
        {
            cuda(Memcpy2DFromArrayAsync(buf.data(), sizeof(float4) * width, array, 0, 0,
                                        sizeof(float4) * width, height,
                                        cudaMemcpyDeviceToHost, stream));
            auto buf_uint8 = std::vector<uint8_t>(4 * width * height);
            for (int j = 0; j < buf.size(); j++)
            {
                buf_uint8[j] = buf[j] * 255;
            }
            std::string fpath = out_dir + "/" + basenames[i] + ".png";
            internal::write_png_file(fpath, buf_uint8.data(), width, height);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cuda(FreeArray(array));
    cuda(StreamDestroy(stream));
    ctx.freeResource();
}
