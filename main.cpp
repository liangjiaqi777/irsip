#include <iostream>
#include <vector>
#include <array>
#include <mutex>
#include <condition_variable>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <arm_neon.h>
#include <chrono>
#include <thread>
#include <functional>
#include <atomic>
#include "cbisp.h"
#include "compress_process.h"
#include <sched.h>
#include <sys/resource.h>

const int g_width = 640;
const int g_height = 512;

struct GuidedResultBuffer {
    std::vector<float> guidedfilter;
    std::vector<float> edge;
    std::vector<uint16_t> raw;

    GuidedResultBuffer(size_t n_pixels) {
        guidedfilter.resize(n_pixels);
        edge.resize(n_pixels);
        raw.resize(n_pixels);
    }
};

std::array<GuidedResultBuffer, 2> g_guided_results = {
    GuidedResultBuffer(g_width * g_height),
    GuidedResultBuffer(g_width * g_height)
};
std::atomic<int> g_active_guided_idx(0);

struct CoeffResultBuffer {
    std::vector<float> a1, b1, a2, b2;
    CoeffResultBuffer(size_t n_pixels) : a1(n_pixels), b1(n_pixels), a2(n_pixels), b2(n_pixels) {}
};

std::array<CoeffResultBuffer, 2> g_coeff_results = {
    CoeffResultBuffer(g_width * g_height),
    CoeffResultBuffer(g_width * g_height)
};

std::array<std::vector<float>, 2> g_mfa_buffers = {
    std::vector<float>(g_width* g_height),
    std::vector<float>(g_width* g_height)
};
std::atomic<int> g_mfa_ready_idx(-1);

static std::mutex              g_sync_mtx;
static std::condition_variable g_sync_cv;

std::atomic<int> g_coeff_active_idx{-1};
std::atomic<int> g_last_guided_ready_idx{-1};
std::array<std::atomic<int>, 2> g_guided_coeff_src = { { -1, -1 } };

static bool read_frame_at(FILE* f, int frame_index, std::vector<uint16_t>& buf) {
    buf.resize(g_width * g_height);
    if (std::fseek(f, static_cast<long>(frame_index) * static_cast<long>(static_cast<size_t>(g_width * g_height) * sizeof(uint16_t)), SEEK_SET) != 0)
        return false;
    size_t n = std::fread(buf.data(), 1, static_cast<size_t>(g_width * g_height) * sizeof(uint16_t), f);
    return n == static_cast<size_t>(g_width * g_height) * sizeof(uint16_t);
}

template<typename... Args>
static void bind_this_thread_to_cpu(Args... cpu_ids) {
    cpu_set_t set;
    CPU_ZERO(&set);

    // 使用C++11的初始化列表和范围for循环，这是最清晰且兼容性最好的方法。
    for (int cpu_id : {cpu_ids...}) {
        CPU_SET(cpu_id, &set);
    }

    int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    if (rc != 0) {
        perror("pthread_setaffinity_np");
    }
}

/**
 * @brief 设置线程的调度策略和优先级
 * 
 * @param policy 调度策略, 例如 SCHED_FIFO, SCHED_RR, SCHED_OTHER
 * @param prio   优先级 (1-99 for SCHED_FIFO and SCHED_RR, 0 for SCHED_OTHER)
 */
static bool set_current_thread_priority(int policy, int prio) {
    struct sched_param sp{};
    sp.sched_priority = prio;
    int rc = pthread_setschedparam(pthread_self(), policy, &sp);
    if (rc != 0) {
        std::fprintf(stderr, "pthread_setschedparam(policy=%d, prio=%d) failed: %s\n",
                     policy, prio, std::strerror(rc));
        return false;
    }
    return true;
}

/**
 * @brief 设置普通线程的nice值
 * 
 * @param nice_val -20 (最高优先级) 到 19 (最低优先级)
 */
static void set_current_thread_nice(int nice_val) {
    int rc = setpriority(PRIO_PROCESS, 0, nice_val);
    if (rc != 0) {
        std::perror("setpriority");
    }
}

static void log_current_sched(const char* tag) {
    int policy{};
    struct sched_param sp{};
    if (pthread_getschedparam(pthread_self(), &policy, &sp) == 0) {
        std::printf("[%s] policy=%s prio=%d\n", tag,
                    policy == SCHED_FIFO ? "SCHED_FIFO" :
                    policy == SCHED_RR   ? "SCHED_RR"   :
                    policy == SCHED_OTHER? "SCHED_OTHER": "UNKNOWN",
                    sp.sched_priority);
    }
}

void save_tif_image(const std::string& filename, const std::vector<float>& data, int width, int height) {
    if (data.empty()) return;

    float min_val = data[0], max_val = data[0];
    for(float val : data) {
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }

    std::vector<unsigned char> uchar_data(data.size());
    float range = max_val - min_val;
    if (range < 1e-6) range = 1.0f;

    for(size_t i = 0; i < data.size(); ++i) {
        uchar_data[i] = static_cast<unsigned char>(((data[i] - min_val) / range) * 255.0f);
    }

    cv::Mat mat_to_save(height, width, CV_8UC1, uchar_data.data());
    cv::imwrite(filename, mat_to_save);
}

void *AlgoProcess(void* p) {
    bind_this_thread_to_cpu(0);

    set_current_thread_priority(SCHED_FIFO, 99);

    log_current_sched("Algo");

    pthread_detach(pthread_self());
    printf("start %s thread, arg: %p\n", __func__, p);

    pthread_detach(pthread_self());
    printf("start %s thread, arg: %p\n", __func__, p);

    const int n_pixels = g_width * g_height;

    std::vector<float> workspace(n_pixels * 3);
    int last_edge_done = -1;

    while (1) {
        auto t_algo0 = std::chrono::high_resolution_clock::now();
        int mfa_idx = -1;
        {
            std::unique_lock<std::mutex> lk(g_sync_mtx);
            g_sync_cv.wait(lk, [] { return g_mfa_ready_idx.load() != -1; });
            mfa_idx = g_mfa_ready_idx.load();
            g_mfa_ready_idx.store(-1);
        }
        // guided filter 计算系数
        const std::vector<float>& mfa_out = g_mfa_buffers[mfa_idx];
        CoeffResultBuffer& coeff = g_coeff_results[mfa_idx];
        guidedfilter_coeffs_two_eps(mfa_out, g_width, g_height, 100.f, 8000.f,
                                    coeff.a1, coeff.b1, coeff.a2, coeff.b2, workspace);
        g_coeff_active_idx.store(mfa_idx, std::memory_order_release);
        
        // 计算edge
        int guided_slot = g_last_guided_ready_idx.load(std::memory_order_acquire);
        if (guided_slot != -1 && guided_slot != last_edge_done) {
            int used_coeff_idx = g_guided_coeff_src[guided_slot].load(std::memory_order_acquire);
            if (used_coeff_idx < 0) used_coeff_idx = guided_slot;

            const CoeffResultBuffer& coeff_used = g_coeff_results[used_coeff_idx];
            const std::vector<float>& mfa_edge = g_mfa_buffers[guided_slot];
            GuidedResultBuffer& target_buffer = g_guided_results[guided_slot];

            const float* __restrict a2p = coeff_used.a2.data();
            const float* __restrict b2p = coeff_used.b2.data();
            const float* __restrict mfp = mfa_edge.data();
            const float* __restrict gfp = target_buffer.guidedfilter.data();
            float* __restrict edgep = target_buffer.edge.data();

            for (int i = 0; i < n_pixels; ++i) {
                edgep[i] = gfp[i] - (a2p[i] * mfp[i] + b2p[i]);
            }

            g_active_guided_idx.store(guided_slot, std::memory_order_release);
            last_edge_done = guided_slot;
        }
        auto t_algo1 = std::chrono::high_resolution_clock::now();
        double t_algo = std::chrono::duration<double, std::milli>(t_algo1 - t_algo0).count();
        std::cout << "Algo process done, time: " << t_algo << " ms." << std::endl;
    }
    return nullptr;
}

void *MainMedia(void* p) {
    bind_this_thread_to_cpu(1);

    set_current_thread_priority(SCHED_FIFO, 99);

    log_current_sched("Main");

    pthread_detach(pthread_self());
    printf("start %s thread, arg: %p\n", __func__, p);

    int width = g_width;
    int height = g_height;
    int n_pixels = width * height;

    // 保存视频测试
    constexpr int SAVE_START_FRAME = 1;          // 起始帧(含)，cnt 从1开始
    constexpr int SAVE_END_FRAME   = 1000;       // 结束帧(含)
    constexpr const char* SAVE_DIR = "/root/qutao/image";
    bool saving_enabled = true;

    // 模拟帧序列
    const char* raw_data = "/root/qutao/scene.dat";
    const char* shut_data = "/root/qutao/shutter.dat";
    FILE* f_raw = std::fopen(raw_data, "rb");
    FILE* f_shutter = std::fopen(shut_data, "rb");
    std::vector<uint16_t> raw_u16(n_pixels), shut_u16(n_pixels);
    std::vector<float> nuc_cur(n_pixels);
    int mfa_buffer_idx = 0;

    // compress param
    int orgscale = 2047;
    std::vector<uint16_t> raw(n_pixels);
    std::vector<uint16_t> raw11(n_pixels);
    std::vector<uint16_t> raw2(n_pixels);
    std::vector<uint32_t> Hist_all(orgscale + 1);
    std::vector<uint32_t> Hist_filtered1(orgscale + 1);
    std::vector<uint32_t> Hist_filtered2(orgscale + 1);
    std::vector<uint16_t> rearrange_table(orgscale + 1);
    std::vector<uint8_t> img_8bit(n_pixels);
    float ratio_low = 0.0f, ratio_high = 0.0f;
    uint8_t img8_min = 0;

    // clahe
    std::vector<uint8_t> clahe_image(n_pixels);
    std::vector<uint8_t> output_image_u8(n_pixels);

    int cnt = 0;
    while (1) {
        cnt++;
        int idx = cnt % 1000;
        int idx2 = cnt % 500;
        if (!read_frame_at(f_raw, idx, raw_u16) || !read_frame_at(f_shutter, idx2, shut_u16)) {
            std::fprintf(stderr, "Read frame failed at index %d\n", idx);
            break;
        }
        auto t_total0 = std::chrono::high_resolution_clock::now();

        // nuc
        auto t_nuc0 = std::chrono::high_resolution_clock::now();
        nuc_apply_neon_scalar_gain(raw_u16.data(), shut_u16.data(), 1.0f, nuc_cur.data(), n_pixels);
        auto t_nuc1 = std::chrono::high_resolution_clock::now();
        double t_nuc = std::chrono::duration<double, std::milli>(t_nuc1 - t_nuc0).count();

        // mfa
        auto t_mfa0 = std::chrono::high_resolution_clock::now();
        mfa_fuse_once_neon(g_mfa_buffers[1 - mfa_buffer_idx].data(), nuc_cur.data(), 
                                g_mfa_buffers[mfa_buffer_idx].data(), n_pixels);
        auto t_mfa1 = std::chrono::high_resolution_clock::now();
        double t_mfa = std::chrono::duration<double, std::milli>(t_mfa1 - t_mfa0).count();

        // guided
        auto t_guided0 = std::chrono::high_resolution_clock::now();
        const int produced_idx = mfa_buffer_idx;
        int coeff_idx = g_coeff_active_idx.load(std::memory_order_acquire);
        if (coeff_idx < 0) coeff_idx = produced_idx;
        {
            const auto& coeff   = g_coeff_results[coeff_idx];
            const auto& mfa_out = g_mfa_buffers[produced_idx];
            GuidedResultBuffer& target_buffer = g_guided_results[produced_idx];
            for (int i = 0; i < n_pixels; ++i) {
                target_buffer.guidedfilter[i] = coeff.a1[i] * mfa_out[i] + coeff.b1[i];
            }
            f32_to_u16_neon(target_buffer.guidedfilter.data(), target_buffer.raw.data(), n_pixels);
            g_guided_coeff_src[produced_idx].store(coeff_idx, std::memory_order_release);
            g_last_guided_ready_idx.store(produced_idx, std::memory_order_release);
        }
        {
            std::lock_guard<std::mutex> lk(g_sync_mtx);
            g_mfa_ready_idx.store(mfa_buffer_idx);
        }
        g_sync_cv.notify_one();
        mfa_buffer_idx = 1 - mfa_buffer_idx;
        int read_idx = g_active_guided_idx.load(std::memory_order_acquire);
        const GuidedResultBuffer& guided_data = g_guided_results[read_idx];
        auto t_guided1 = std::chrono::high_resolution_clock::now(); 
        double ms_guided = std::chrono::duration<double, std::milli>(t_guided1 - t_guided0).count();

        // hist
        auto t_hist0 = std::chrono::high_resolution_clock::now();
        compress_process(guided_data.raw.data(), raw11.data(), raw2.data(),
            Hist_all.data(), Hist_filtered1.data(), Hist_filtered2.data(),
            rearrange_table.data(), img_8bit.data(), height, width, orgscale, 80, 16, 235, 
            ratio_low, ratio_high, img8_min);
        auto t_hist1 = std::chrono::high_resolution_clock::now();
        double t_hist = std::chrono::duration<double, std::milli>(t_hist1 - t_hist0).count();

        // clahe
        auto t_clahe0 = std::chrono::high_resolution_clock::now();
        clahe8u(img_8bit.data(), clahe_image.data(), width, height, 2.0);
        for (int i = 0; i < n_pixels; i++) {
            output_image_u8[i] = static_cast<uint8_t>(std::min(255, std::max(0, clahe_image[i] + static_cast<int>(guided_data.edge[i]))));
        }

        auto t_clahe1 = std::chrono::high_resolution_clock::now();
        double t_clahe = std::chrono::duration<double, std::milli>(t_clahe1 - t_clahe0).count();

        auto t_total1 = std::chrono::high_resolution_clock::now(); 
        double ms_total = std::chrono::duration<double, std::milli>(t_total1 - t_total0).count();
        // if (cnt % 30 == 0) {
            std::cout 
                << "Frame " << cnt
                << ": Total " << ms_total << " ms."
                << " NUC " << t_nuc << " ms."
                << " MFA " << t_mfa << " ms."
                << " Guided " << ms_guided << " ms."
                << " Hist " << t_hist << " ms."
                << " CLAHE " << t_clahe << " ms."
                << std::endl;
        // }

        // if (saving_enabled && cnt >= SAVE_START_FRAME && cnt <= SAVE_END_FRAME) {
        //     char filename[256];
        //     std::snprintf(filename, sizeof(filename), "%s/frame_%06d.tif", SAVE_DIR, cnt);
        //     cv::Mat gray(height, width, CV_8UC1, output_image_u8.data());
        //     std::vector<int> params{
        //         cv::IMWRITE_TIFF_COMPRESSION, 5   // 使用 LZW 无损压缩
        //     };
        //     if (!cv::imwrite(filename, gray, params)) {
        //         std::fprintf(stderr, "imwrite tif failed: %s\n", filename);
        //     }else  if (cnt == SAVE_START_FRAME) {
        //         std::printf("Start saving TIFF frames at %d -> %s\n", cnt, filename);
        //     } else if (cnt == SAVE_END_FRAME) {
        //         std::printf("Reached end frame %d, last file %s\n", cnt, filename);
        //     }
        // }
        // if (saving_enabled && cnt > SAVE_END_FRAME) {
        //     saving_enabled = false; // 之后继续处理但不再保存
        // }

        constexpr double TARGET_FRAME_DURATION_MS = 20.0;
        double sleep_duration_ms = TARGET_FRAME_DURATION_MS - ms_total;
        if (sleep_duration_ms > 0) {
            std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(sleep_duration_ms));
        }
    }
    return nullptr;
}

int main(int argc, char* argv[]) {

    pthread_t algo, main;
    pthread_create(&algo, nullptr, AlgoProcess, nullptr);
    pthread_create(&main, nullptr, MainMedia, nullptr);

    pthread_join(algo, nullptr);
    pthread_join(main, nullptr);

    return 0;
}

// auto start_time = std::chrono::high_resolution_clock::now();
// auto end_time = std::chrono::high_resolution_clock::now();
// std::chrono::duration<double, std::milli> duration = end_time - start_time;
// std::cout << "guidedfilter_self: " << duration.count() << " ms" << std::endl;

// // Save the output as a raw file
// FILE *output_file = fopen("/root/qutao/lou2_output.raw", "wb");
// if (!output_file) {
//     std::cerr << "Error opening output file." << std::endl;
//     return -1;
// }
// fwrite(output_image.data(), sizeof(float), width*height, output_file);
// fclose(output_file);