#include <cstdint>
#include <vector>    
#include <array>     
#include <numeric>   
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <arm_neon.h>
#include <cmath>

#include <chrono>
#include <iostream>

/**
 * @brief NUC
 * 
 * @param scene16   输入原始图像指针
 * @param shutter16 输入原始挡板指针
 * @param gain      增益值
 * @param out32f    输出校正图像指针
 * @param n         像素数量
 */

void nuc_apply_neon_scalar_gain(const uint16_t* __restrict scene16,
                                const uint16_t* __restrict shutter16,
                                float gain,
                                float* __restrict image_out,
                                size_t n)
{
    const uint16x8_t v_mask = vdupq_n_u16(0x3FFF);
    const float32x4_t v_gain = vdupq_n_f32(gain);
    const float32x4_t v_offset = vdupq_n_f32(8192.0f);
    size_t i = 0;

    for (; i + 8 <= n; i += 8) {
        // 1. 加载 8 个 uint16_t 数据
        uint16x8_t v_scene_u16 = vld1q_u16(scene16 + i);
        uint16x8_t v_shutter_u16 = vld1q_u16(shutter16 + i);

        // 2. 应用掩码 & 0x3FFF
        v_scene_u16 = vandq_u16(v_scene_u16, v_mask);
        v_shutter_u16 = vandq_u16(v_shutter_u16, v_mask);

        // 3. 将 uint16 扩展为 int32 (为了后续转换为 float)
        //    vmovl_u16 将 uint16x8_t 的低4个和高4个元素分别扩展为两个 int32x4_t
        int32x4_t v_scene_s32_low  = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_scene_u16)));
        int32x4_t v_scene_s32_high = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_scene_u16)));
        int32x4_t v_shutter_s32_low  = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_shutter_u16)));
        int32x4_t v_shutter_s32_high = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_shutter_u16)));

        // 4. 将 int32 转换为 float32
        float32x4_t v_scene_f32_low  = vcvtq_f32_s32(v_scene_s32_low);
        float32x4_t v_scene_f32_high = vcvtq_f32_s32(v_scene_s32_high);
        float32x4_t v_shutter_f32_low  = vcvtq_f32_s32(v_shutter_s32_low);
        float32x4_t v_shutter_f32_high = vcvtq_f32_s32(v_shutter_s32_high);

        // 5. 执行减法
        float32x4_t v_diff_low  = vsubq_f32(v_scene_f32_low,  v_shutter_f32_low);
        float32x4_t v_diff_high = vsubq_f32(v_scene_f32_high, v_shutter_f32_high);

        // 6. 乘以增益
        float32x4_t v_mul_low  = vmulq_f32(v_diff_low,  v_gain);
        float32x4_t v_mul_high = vmulq_f32(v_diff_high, v_gain);

        // 7. 加上偏移量 8192.0f
        float32x4_t v_out_low  = vaddq_f32(v_mul_low, v_offset);
        float32x4_t v_out_high = vaddq_f32(v_mul_high, v_offset);

        vst1q_f32(image_out + i,     v_out_low);
        vst1q_f32(image_out + i + 4, v_out_high);
    }

    for (; i < n; ++i) {
        image_out[i] = gain * (static_cast<float>(scene16[i] & 0x3FFF) - static_cast<float>(shutter16[i] & 0x3FFF)) + 8192.0f;
    }
}

/**
 * @brief MFA
 *
 */
 
void mfa_fuse_once_neon(const float* __restrict prev_frame,
                        const float* __restrict cur_frame,
                        float* __restrict out_frame,
                        size_t n) {
    const float32x4_t v_four     = vdupq_n_f32(4.0f);
    const float32x4_t v_eighteen = vdupq_n_f32(18.0f);
    const float32x4_t v_one      = vdupq_n_f32(1.0f);
    const float32x4_t v_zero     = vdupq_n_f32(0.0f);

    // 多项式系数 (3~4阶足够)
    const float32x4_t c3 = vdupq_n_f32( 2.1967e-03f);
    const float32x4_t c2 = vdupq_n_f32(-2.7375e-02f);
    const float32x4_t c1 = vdupq_n_f32( 1.4453e-01f);
    const float32x4_t c0 = vdupq_n_f32( 7.2116e-01f);

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vp = vld1q_f32(prev_frame + i);
        float32x4_t vc = vld1q_f32(cur_frame  + i);
        float32x4_t vd = vabdq_f32(vc, vp);

        // Horner 展开
        float32x4_t poly = vfmaq_f32(c2, c3, vd);
        poly = vfmaq_f32(c1, poly, vd);
        poly = vfmaq_f32(c0, poly, vd);

        float32x4_t va = poly;
        va = vbslq_f32(vcgeq_f32(vd, v_eighteen), v_zero, va);
        va = vbslq_f32(vcltq_f32(vd, v_four), v_one, va);

        // out = (cur + a*prev)/(1+a)
        float32x4_t num = vfmaq_f32(vc, va, vp);
        float32x4_t den = vaddq_f32(v_one, va);

        float32x4_t rec = vrecpeq_f32(den);
        rec = vmulq_f32(vrecpsq_f32(den, rec), rec);
        float32x4_t vo  = vmulq_f32(num, rec);

        vst1q_f32(out_frame + i, vo);
    }

    // 尾部处理
    for (; i < n; i++) {
        float d = fabsf(cur_frame[i] - prev_frame[i]);
        float a;
        if (d < 4.f) 
            a = 1.f;
        else if (d >= 18.f) 
            a = 0.f;
        else 
            a = ((c3[0]*d + c2[0])*d + c1[0])*d + c0[0];
        out_frame[i] = (cur_frame[i] + a * prev_frame[i]) / (1.f + a);
    }
}

void BoxFilterBetterNeonAssemblyV2(const float* __restrict Src, 
                                   float* __restrict Dest,
                                   int Width, int Height, int KernelSize) {
    if (KernelSize != 3) {
        return;
    }

    int OutWidth = Width - KernelSize + 1;   // = Width - 2
    int OutHeight = Height - KernelSize + 1; // = Height - 2
    
    int i = 0;
    for (; i + 1 < OutHeight; i += 2) { // 每次处理2行
        const float* p_r0 = Src + i * Width;
        const float* p_r1 = Src + (i + 1) * Width;
        const float* p_r2 = Src + (i + 2) * Width;
        const float* p_r3 = Src + (i + 3) * Width;
        
        float* p_outptr  = Dest + (i + 1) * Width + 1;
        float* p_outptr2 = Dest + (i + 2) * Width + 1;

        int nn = OutWidth >> 2;
        int remain = OutWidth - (nn << 2);

        if (nn > 0) { // 每次循环处理8个像素(2行 x 4列), 计算两次
            asm volatile(
                "mov x10, %[nn]\n" // 循环次数
                "0:\n"
                
                /* 
                 * 预加载数据
                 * prfm: 将数据预加载到L1缓存中
                 * pldl1keep: pld为预取, l1目标为L1缓存, keep为数据加载后多次使用
                 * %[r0]: 基地址，即r0指向的内存地址
                 * #192: 预加载的字节数
                 * 
                 * ldp: load pair, 用于加载两个寄存器
                 * q0, q1: 目标寄存器，q表示128位寄存器
                 * [%[r0]]: 从r0指向的地址加载256位数据到q0和q1寄存器
                 */
                "prfm pldl1keep, [%[r0], #192]\n"   // 加载数据到L1缓存
                "ldp q0, q1, [%[r0]]\n"
                "prfm pldl1keep, [%[r1], #192]\n"
                "ldp q2, q3, [%[r1]]\n"
                "prfm pldl1keep, [%[r2], #192]\n"
                "ldp q4, q5, [%[r2]]\n"
                "prfm pldl1keep, [%[r3], #192]\n"
                "ldp q6, q7, [%[r3]]\n"

                /*
                 * 水平求和
                 * 采用ext和fadd指令计算每一行滑动的3个像素和
                 * 计算得到r0行的水平和
                 */

                /*
                 * v8.16b: 目标寄存器
                 * v0.16b: 源寄存器1
                 * v1.16b: 源寄存器2
                 *  .16b表示每一个寄存器包含16个8位元素(4个float32)
                 * eg: 
                    v0: [0, 1, 2, 3]
                    v1: [4, 5, 6, 7]

                    v8: [1, 2, 3, 4]  v0移1个32
                    v9: [2, 3, 4, 5]  v0移2个32
                 */
                "ext v8.16b, v0.16b, v1.16b, #4\n"   
                "ext v9.16b, v0.16b, v1.16b, #8\n"  
                
                /*
                 * v16.4s: 目标寄存器
                 * v0.4s: 源寄存器1
                 * v8.4s: 源寄存器2
                 *  .4s表示每一个寄存器包含4个32位元素(float)
                 * eg: 
                 *  v0: [0, 1, 2, 3]
                 *  v8: [1, 2, 3, 4]
                 *  v9: [2, 3, 4, 5]
                 *  v16:[0+1+2, 1+2+3, 2+3+4, 3+4+5]  // r0行
                 */
                "fadd v16.4s, v0.4s, v8.4s\n"
                "fadd v16.4s, v16.4s, v9.4s\n"          

                /*
                 * v17:[0+1+2, 1+2+3, 2+3+4, 3+4+5]   // r1行
                 */
                "ext v10.16b, v2.16b, v3.16b, #4\n"  
                "ext v11.16b, v2.16b, v3.16b, #8\n"  
                "fadd v17.4s, v2.4s, v10.4s\n"
                "fadd v17.4s, v17.4s, v11.4s\n"

                /*
                 * v18:[0+1+2, 1+2+3, 2+3+4, 3+4+5]   // r2行
                 */
                "ext v8.16b, v4.16b, v5.16b, #4\n"   
                "ext v9.16b, v4.16b, v5.16b, #8\n"   
                "fadd v18.4s, v4.4s, v8.4s\n"
                "fadd v18.4s, v18.4s, v9.4s\n"

                /*
                 * v19:[0+1+2, 1+2+3, 2+3+4, 3+4+5]   // r3行
                 */
                "ext v10.16b, v6.16b, v7.16b, #4\n"  
                "ext v11.16b, v6.16b, v7.16b, #8\n"  
                "fadd v19.4s, v6.4s, v10.4s\n"
                "fadd v19.4s, v19.4s, v11.4s\n"

                /*
                 * 垂直求和
                 * v20: [r0+r1+r2]
                 */
                "fadd v20.4s, v16.4s, v17.4s\n"         
                "fadd v20.4s, v20.4s, v18.4s\n"         

                /* 
                 * v21: [r1+r2+r3]
                 */
                "fadd v21.4s, v17.4s, v18.4s\n"        
                "fadd v21.4s, v21.4s, v19.4s\n"         

                /*
                 * 存储结果
                 * 将计算结果从NEON寄存器写回内存
                 * %[outptr]和[outptr2]是输出指针，#16表示每次写入16字节(128位)
                 * q20: 源寄存器1
                 */
                "str q20, [%[outptr]], #16\n"   // 拷贝r0+r1+r2行像素和到输出
                "str q21, [%[outptr2]], #16\n"  // 拷贝r1+r2+r3行像素和到输出
                
                /* 
                 * 更新指针, 下次循环
                 * 取出r0指针的当前地址值加上16然后将结果写回到r0指针
                 * 加16字节: 每次处理128位数据
                 */
                "add %[r0], %[r0], #16\n"
                "add %[r1], %[r1], #16\n"
                "add %[r2], %[r2], #16\n"
                "add %[r3], %[r3], #16\n"
                
                /*
                 * 循环控制
                 * subs: sub执行减法, s减法结果为0, Z标志为1
                 * x10 = x10 - 1
                 */
                "subs x10, x10, #1\n"
                "bne 0b\n"

                : [r0] "+r"(p_r0), [r1] "+r"(p_r1), [r2] "+r"(p_r2), [r3] "+r"(p_r3),
                  [outptr] "+r"(p_outptr), [outptr2] "+r"(p_outptr2)
                : [nn] "r"((long)nn)
                : "cc", "memory", "x10", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19", "v20", "v21"
            );
        }

        for(; remain > 0; remain--){
            float h_sum0 = p_r0[0] + p_r0[1] + p_r0[2];
            float h_sum1 = p_r1[0] + p_r1[1] + p_r1[2];
            float h_sum2 = p_r2[0] + p_r2[1] + p_r2[2];
            *p_outptr = h_sum0 + h_sum1 + h_sum2;

            float h_sum3 = p_r3[0] + p_r3[1] + p_r3[2];
            *p_outptr2 = h_sum1 + h_sum2 + h_sum3;

            p_r0++; p_r1++; p_r2++; p_r3++;
            p_outptr++; p_outptr2++;
        }
    }
}

void boxfilter(const float* __restrict I,
               float* __restrict result,
               int width, int height, int r) {
    int n = width * height;
    float* temp = new float[n](); // 临时数组，用于存储中间结果

    // 计算积分图的 Lambda 函数
    auto compute_integral = [](const float* __restrict src,
                               float* __restrict integral,
                               int w, int h) {
        integral[0] = src[0];
        for (int x = 1; x < w; ++x) 
            integral[x] = integral[x - 1] + src[x];
        
        for (int y = 1; y < h; ++y) {
            float row_sum = 0;
            for (int x = 0; x < w; ++x) {
                row_sum += src[y * w + x];
                integral[y * w + x] = integral[(y - 1) * w + x] + row_sum;
            }
        }
    };

    // 分配积分图数组
    float* integral_horiz = new float[n]();
    compute_integral(I, integral_horiz, width, height);

    // 水平方向的均值滤波
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int x1 = std::max(x - r, 0);
            int x2 = std::min(x + r, width - 1);

            float sum = integral_horiz[y * width + x2];
            if (x1 > 0) sum -= integral_horiz[y * width + x1 - 1];
            int count = x2 - x1 + 1;

            temp[y * width + x] = sum / count;
        }
    }

    // 垂直方向的积分图
    float* integral_vert = new float[n]();
    compute_integral(temp, integral_vert, width, height);

    // 垂直方向的均值滤波
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int y1 = std::max(y - r, 0);
            int y2 = std::min(y + r, height - 1);

            float sum = integral_vert[y2 * width + x];
            if (y1 > 0) sum -= integral_vert[(y1 - 1) * width + x];
            int count = y2 - y1 + 1;

            result[y * width + x] = sum / count;
        }
    }

    // 释放动态分配的内存
    delete[] temp;
    delete[] integral_horiz;
    delete[] integral_vert;
}

std::vector<float> guidedfilter_self(const std::vector<float>& p,
                                     int width, int height,
                                     float eps,
                                     std::vector<float>& workspace) {
    const int n = width * height;

    float* __restrict p_squared = workspace.data();
    float* __restrict mean_p    = p_squared + n;
    float* __restrict mean_p2   = mean_p + n;

    const float* __restrict p_ptr = p.data();
    std::vector<float> q(n);
    float* __restrict q_ptr = q.data();

    for (int i = 0; i < n; i += 4) {
        float32x4_t p_vec = vld1q_f32(p_ptr + i);
        vst1q_f32(p_squared + i, vmulq_f32(p_vec, p_vec));
    }

    BoxFilterBetterNeonAssemblyV2(p.data(), mean_p, width, height, 3);
    BoxFilterBetterNeonAssemblyV2(p_squared, mean_p2, width, height, 3);

    if (width >= 3 && height >= 3) {
        // 顶/底行
        std::memcpy(mean_p, mean_p + width, sizeof(float) * width);
        std::memcpy(mean_p + (height - 1) * width, mean_p + (height - 2) * width, sizeof(float) * width);
        std::memcpy(mean_p2, mean_p2 + width, sizeof(float) * width);
        std::memcpy(mean_p2 + (height - 1) * width, mean_p2 + (height - 2) * width, sizeof(float) * width);
        // 左/右列
        for (int y = 0; y < height; ++y) {
            float* r1 = mean_p  + y * width;
            float* r2 = mean_p2 + y * width;
            r1[0] = r1[1];
            r1[width - 1] = r1[width - 2];
            r2[0] = r2[1];
            r2[width - 1] = r2[width - 2];
        }
    }

    const float norm_val = 1.0f / 9.0f; 
    const float32x4_t v_norm = vdupq_n_f32(norm_val);
    const float32x4_t v_eps = vdupq_n_f32(eps);
    const float32x4_t v_one = vdupq_n_f32(1.0f);
    const float32x4_t v_zero = vdupq_n_f32(0.0f);

    for (int i = 0; i < n; i += 4) {
        float32x4_t v_mean_p = vld1q_f32(mean_p + i);
        float32x4_t v_mean_p2 = vld1q_f32(mean_p2 + i);
        
        // var_p = mean_p2 * norm - (mean_p * norm)^2
        float32x4_t v_mean_p_norm = vmulq_f32(v_mean_p, v_norm);
        float32x4_t v_var_p = vfmsq_f32(vmulq_f32(v_mean_p2, v_norm), v_mean_p_norm, v_mean_p_norm); // Fused Multiply-Subtract

        // a = var_p / (var_p + eps)
        // 使用倒数近似来优化除法
        float32x4_t v_denom = vaddq_f32(v_var_p, v_eps);
        float32x4_t v_inv_denom = vrecpeq_f32(v_denom); // 近似倒数
        v_inv_denom = vmulq_f32(vrecpsq_f32(v_denom, v_inv_denom), v_inv_denom); // 牛顿-拉夫森迭代
        float32x4_t v_a = vmulq_f32(v_var_p, v_inv_denom);
        
        // b = mean_p_norm * (1.0 - a)
        float32x4_t v_b = vmulq_f32(v_mean_p_norm, vsubq_f32(v_one, v_a));

        // q = a * p + b
        float32x4_t v_p = vld1q_f32(p_ptr + i);
        float32x4_t v_q = vfmaq_f32(v_b, v_a, v_p); // Fused Multiply-Add

        vst1q_f32(q.data() + i, v_q);
    }

    return q;
}

#if 1
// void guidedfilter_coeffs_two_eps(const std::vector<float>& p, int width, int height,
//                                  float eps1, float eps2,
//                                  std::vector<float>& a1, std::vector<float>& b1,
//                                  std::vector<float>& a2, std::vector<float>& b2,
//                                  std::vector<float>& workspace)
// {
//     const int n = width * height;

//     float* __restrict p_squared = workspace.data();
//     float* __restrict mean_p    = p_squared + n;
//     float* __restrict mean_p2   = mean_p + n;

//     const float* __restrict p_ptr = p.data();
//     float* __restrict a1_ptr = a1.data();
//     float* __restrict b1_ptr = b1.data();
//     float* __restrict a2_ptr = a2.data();
//     float* __restrict b2_ptr = b2.data();

//     for (int i = 0; i < n; i += 4) {
//         float32x4_t v = vld1q_f32(p_ptr + i);
//         vst1q_f32(p_squared + i, vmulq_f32(v, v));
//     }

//     BoxFilterBetterNeonAssemblyV2(p.data(),  mean_p,  width, height, 3);
//     BoxFilterBetterNeonAssemblyV2(p_squared, mean_p2, width, height, 3);

//     if (width >= 3 && height >= 3) {
//         // 顶/底行
//         std::memcpy(mean_p, mean_p + width, sizeof(float) * width);
//         std::memcpy(mean_p + (height - 1) * width, mean_p + (height - 2) * width, sizeof(float) * width);
//         std::memcpy(mean_p2, mean_p2 + width, sizeof(float) * width);
//         std::memcpy(mean_p2 + (height - 1) * width, mean_p2 + (height - 2) * width, sizeof(float) * width);
//         // 左/右列
//         for (int y = 0; y < height; ++y) {
//             float* r1 = mean_p  + y * width;
//             float* r2 = mean_p2 + y * width;
//             r1[0] = r1[1];
//             r1[width - 1] = r1[width - 2];
//             r2[0] = r2[1];
//             r2[width - 1] = r2[width - 2];
//         }
//     }

//     const float norm_val = 1.0f / 9.0f;
//     const float32x4_t v_norm = vdupq_n_f32(norm_val);
//     const float32x4_t v_eps1 = vdupq_n_f32(eps1);
//     const float32x4_t v_eps2 = vdupq_n_f32(eps2);
//     const float32x4_t v_one = vdupq_n_f32(1.0f);
//     const float32x4_t v_zero = vdupq_n_f32(0.0f);
//     // a2系数调整常量
//     const float32x4_t v_035 = vdupq_n_f32(0.35f);
//     const float32x4_t v_06 = vdupq_n_f32(0.6f);
//     const float32x4_t v_0568 = vdupq_n_f32(0.568f);
//     const float32x4_t v_1512 = vdupq_n_f32(0.1512f);
//     const float32x4_t v_085 = vdupq_n_f32(0.85f);
//     const float32x4_t v_018 = vdupq_n_f32(0.018f);

//     for (int i = 0; i < n; i += 4) {
//         // 1. 加载均值数据
//         float32x4_t v_mean_p = vld1q_f32(mean_p + i);
//         float32x4_t v_mean_p2 = vld1q_f32(mean_p2 + i);

//         // 2. 计算归一化均值和方差
//         float32x4_t v_mean_p_norm = vmulq_f32(v_mean_p, v_norm);
//         float32x4_t v_var_p = vfmsq_f32(
//             vmulq_f32(v_mean_p2, v_norm),
//             v_mean_p_norm, v_mean_p_norm
//         );
//         v_var_p = vmaxq_f32(v_var_p, v_zero);

//         // 3. 计算 a1 和 b1 (基础层)
//         float32x4_t v_denom1 = vaddq_f32(v_var_p, v_eps1);
//         float32x4_t v_inv1 = vrecpeq_f32(v_denom1);
//         v_inv1 = vmulq_f32(vrecpsq_f32(v_denom1, v_inv1), v_inv1);
//         float32x4_t v_a1 = vmulq_f32(v_var_p, v_inv1);
//         float32x4_t v_b1 = vmulq_f32(v_mean_p_norm, vsubq_f32(v_one, v_a1));
//         vst1q_f32(a1_ptr + i, v_a1);
//         vst1q_f32(b1_ptr + i, v_b1);

//         // 4. 计算原始 a2 (细节层)
//         float32x4_t v_denom2 = vaddq_f32(v_var_p, v_eps2);
//         float32x4_t v_inv2 = vrecpeq_f32(v_denom2);
//         v_inv2 = vmulq_f32(vrecpsq_f32(v_denom2, v_inv2), v_inv2);
//         float32x4_t v_a2 = vmulq_f32(v_var_p, v_inv2);

//         // 5. 根据 a1 值调整 a2 (使用向量化条件)
//         uint32x4_t mask_lt_035 = vcltq_f32(v_a1, v_035);  // a1 < 0.35
//         uint32x4_t mask_lt_06 = vcltq_f32(v_a1, v_06);    // a1 < 0.6
//         uint32x4_t mask_mid = vandq_u32(vmvnq_u32(mask_lt_035), mask_lt_06); // 0.35 <= a1 < 0.6
//         uint32x4_t mask_ge_06 = vmvnq_u32(mask_lt_06);    // a1 >= 0.6

//         // 计算每个分段的值
//         float32x4_t val1 = v_a1;  // a1 < 0.35: a2 = a1
//         float32x4_t val2 = vaddq_f32(vmulq_f32(v_a2, v_0568), v_1512);  // 0.35 <= a1 < 0.6: a2 = 0.568*a2 + 0.1512
//         float32x4_t val3 = vsubq_f32(vmulq_f32(v_a2, v_085), v_018);    // a1 >= 0.6: a2 = 0.85*a2 - 0.018

//         // 使用掩码选择合适的值
//         v_a2 = vbslq_f32(mask_lt_035, val1, 
//                 vbslq_f32(mask_mid, val2, val3));

//         // 6. 计算 b2 并存储
//         float32x4_t v_b2 = vmulq_f32(v_mean_p_norm, vsubq_f32(v_one, v_a2));
//         vst1q_f32(a2_ptr + i, v_a2);
//         vst1q_f32(b2_ptr + i, v_b2);
//     }
// }
void guidedfilter_coeffs_two_eps(const std::vector<float>& p, int width, int height,
                                 float eps1, float eps2,
                                 std::vector<float>& a1, std::vector<float>& b1,
                                 std::vector<float>& a2, std::vector<float>& b2,
                                 std::vector<float>& workspace)
{
    const int n = width * height;

    float* __restrict p_squared = workspace.data();
    float* __restrict mean_p    = p_squared + n;
    float* __restrict mean_p2   = mean_p + n;

    const float* __restrict p_ptr = p.data();
    float* __restrict a1_ptr = a1.data();
    float* __restrict b1_ptr = b1.data();
    float* __restrict a2_ptr = a2.data();
    float* __restrict b2_ptr = b2.data();

    // 1. 计算 p^2 (向量化)
    for (int i = 0; i < n; i += 4) {
        float32x4_t v = vld1q_f32(p_ptr + i);
        vst1q_f32(p_squared + i, vmulq_f32(v, v));
    }

    // 2. Box Filter (汇编优化版)
    BoxFilterBetterNeonAssemblyV2(p.data(),  mean_p,  width, height, 3);
    BoxFilterBetterNeonAssemblyV2(p_squared, mean_p2, width, height, 3);

    // 3. 边界填充
    if (width >= 3 && height >= 3) {
        std::memcpy(mean_p, mean_p + width, sizeof(float) * width);
        std::memcpy(mean_p + (height - 1) * width, mean_p + (height - 2) * width, sizeof(float) * width);
        std::memcpy(mean_p2, mean_p2 + width, sizeof(float) * width);
        std::memcpy(mean_p2 + (height - 1) * width, mean_p2 + (height - 2) * width, sizeof(float) * width);
        for (int y = 0; y < height; ++y) {
            float* r1 = mean_p  + y * width;
            float* r2 = mean_p2 + y * width;
            r1[0] = r1[1]; r1[width - 1] = r1[width - 2];
            r2[0] = r2[1]; r2[width - 1] = r2[width - 2];
        }
    }

    // 4. 计算系数 (全 NEON 优化)
    const float norm_val = 1.0f / 9.0f;
    const float32x4_t v_norm = vdupq_n_f32(norm_val);
    const float32x4_t v_eps1 = vdupq_n_f32(eps1);
    const float32x4_t v_eps2 = vdupq_n_f32(eps2);
    const float32x4_t v_one  = vdupq_n_f32(1.0f);
    const float32x4_t v_zero = vdupq_n_f32(0.0f);
    
    // a2 调整阈值
    const float32x4_t v_035 = vdupq_n_f32(0.35f);
    const float32x4_t v_060 = vdupq_n_f32(0.60f);
    const float32x4_t v_0568 = vdupq_n_f32(0.568f);
    const float32x4_t v_1512 = vdupq_n_f32(0.1512f);
    const float32x4_t v_085  = vdupq_n_f32(0.85f);
    const float32x4_t v_018  = vdupq_n_f32(0.018f);

    for (int i = 0; i < n; i += 4) {
        float32x4_t v_mp = vld1q_f32(mean_p + i);
        float32x4_t v_mp2 = vld1q_f32(mean_p2 + i);

        // var_p = (mean_p2 * norm) - (mean_p * norm)^2
        float32x4_t v_mp_n = vmulq_f32(v_mp, v_norm);
        float32x4_t v_var = vfmsq_f32(vmulq_f32(v_mp2, v_norm), v_mp_n, v_mp_n);
        v_var = vmaxq_f32(v_var, v_zero); // 保证非负

        // 计算 a1, b1
        // 使用牛顿迭代法求倒数: 1/(var+eps)
        float32x4_t v_den1 = vaddq_f32(v_var, v_eps1);
        float32x4_t v_inv1 = vrecpeq_f32(v_den1);
        v_inv1 = vmulq_f32(vrecpsq_f32(v_den1, v_inv1), v_inv1);
        
        float32x4_t v_a1 = vmulq_f32(v_var, v_inv1);
        float32x4_t v_b1 = vmulq_f32(v_mp_n, vsubq_f32(v_one, v_a1));
        
        vst1q_f32(a1_ptr + i, v_a1);
        vst1q_f32(b1_ptr + i, v_b1);

        // 计算 a2 (原始)
        float32x4_t v_den2 = vaddq_f32(v_var, v_eps2);
        float32x4_t v_inv2 = vrecpeq_f32(v_den2);
        v_inv2 = vmulq_f32(vrecpsq_f32(v_den2, v_inv2), v_inv2);
        float32x4_t v_a2 = vmulq_f32(v_var, v_inv2);

        // 调整 a2 (向量化分支消除)
        uint32x4_t m_lt_035 = vcltq_f32(v_a1, v_035);
        uint32x4_t m_lt_060 = vcltq_f32(v_a1, v_060);
        
        // 分段计算
        float32x4_t v_mid = vaddq_f32(vmulq_f32(v_a2, v_0568), v_1512); // 0.35~0.6
        float32x4_t v_hi  = vsubq_f32(vmulq_f32(v_a2, v_085),  v_018);  // >0.6
        
        // 选择逻辑: a1 < 0.35 ? a2 : (a1 < 0.6 ? mid : hi)
        v_a2 = vbslq_f32(m_lt_035, v_a2, vbslq_f32(m_lt_060, v_mid, v_hi));

        // 计算 b2
        float32x4_t v_b2 = vmulq_f32(v_mp_n, vsubq_f32(v_one, v_a2));
        
        vst1q_f32(a2_ptr + i, v_a2);
        vst1q_f32(b2_ptr + i, v_b2);
    }
}

// 2. 【新增】Edge 计算专用函数
// 公式: edge = gfp - (a2 * mfp + b2)
void guided_filter_compute_edge_neon(const float* gfp, 
                                     const float* a2p, 
                                     const float* mfp, 
                                     const float* b2p, 
                                     float* edgep, 
                                     int n_pixels) 
{
    int i = 0;
    // 主循环：每次处理 4 个像素
    for (; i <= n_pixels - 4; i += 4) {
        float32x4_t v_g  = vld1q_f32(gfp + i);
        float32x4_t v_a2 = vld1q_f32(a2p + i);
        float32x4_t v_m  = vld1q_f32(mfp + i);
        float32x4_t v_b2 = vld1q_f32(b2p + i);
        
        // detail = a2 * mfp + b2
        float32x4_t v_detail = vfmaq_f32(v_b2, v_a2, v_m);
        
        // edge = gfp - detail
        float32x4_t v_edge = vsubq_f32(v_g, v_detail);
        
        vst1q_f32(edgep + i, v_edge);
    }
    
    // 处理剩余像素
    for (; i < n_pixels; ++i) {
        edgep[i] = gfp[i] - (a2p[i] * mfp[i] + b2p[i]);
    }
}
#else
void guidedfilter_coeffs_two_eps(
    const std::vector<float>& p, int width, int height,
    float eps1, float eps2,
    std::vector<float>& a1, std::vector<float>& b1,
    std::vector<float>& a2, std::vector<float>& b2,
    std::vector<float>& workspace)
{
    const int n = width * height;

    float* p_squared = workspace.data();
    float* mean_p    = p_squared + n;
    float* mean_p2   = mean_p + n;

    const float* p_ptr = p.data();
    const int n_vec = (n / 4) * 4;
    for (int i = 0; i < n; i += 4) {
        float32x4_t v = vld1q_f32(p_ptr + i);
        vst1q_f32(p_squared + i, vmulq_f32(v, v));
    }

    BoxFilterBetterNeonAssemblyV2(p.data(),  mean_p,  width, height, 3);
    BoxFilterBetterNeonAssemblyV2(p_squared, mean_p2, width, height, 3);

    if (width >= 3 && height >= 3) {
        // 顶/底行
        std::memcpy(mean_p, mean_p + width, sizeof(float) * width);
        std::memcpy(mean_p + (height - 1) * width, mean_p + (height - 2) * width, sizeof(float) * width);
        std::memcpy(mean_p2, mean_p2 + width, sizeof(float) * width);
        std::memcpy(mean_p2 + (height - 1) * width, mean_p2 + (height - 2) * width, sizeof(float) * width);
        // 左/右列
        for (int y = 0; y < height; ++y) {
            float* r1 = mean_p  + y * width;
            float* r2 = mean_p2 + y * width;
            r1[0] = r1[1];
            r1[width - 1] = r1[width - 2];
            r2[0] = r2[1];
            r2[width - 1] = r2[width - 2];
        }
    }

    const float norm_val = 1.0f / 9.0f;
    float* a1_ptr = a1.data(); float* b1_ptr = b1.data();
    float* a2_ptr = a2.data(); float* b2_ptr = b2.data();
    for (int i = 0; i < n; ++i) {
        float meanp  = mean_p[i]  * norm_val;
        float meanp2 = mean_p2[i] * norm_val;
        float varp   = std::max(0.0f, meanp2 - meanp * meanp);

        // 基础层
        float a_1 = varp / (varp + eps1);
        float b_1 = meanp * (1.0f - a_1);
        a1_ptr[i] = a_1; b1_ptr[i] = b_1;

        // 细节掩模层
        float a_2 = varp / (varp + eps2);

        // a2系数调整
        if (a_1 < 0.35f) {
            a_2 = a_1;  
        } else if (a_1 < 0.6f) {
            a_2 = 0.568f * a_2 + 0.1512f;  // 中频区域
        } else {
            a_2 = 0.85f * a_2 - 0.018f;  // 高频区域
        }

        float b_2 = meanp * (1.0f - a_2);
        a2_ptr[i] = a_2; b2_ptr[i] = b_2;
    }
}
#endif

/**
 * @brief Fused Multiply-Add
 * 
 * @param a 乘数1
 * @param b 乘数2
 * @param c 加数
 */
void guided_filter_fma_neon_asm(const float* __restrict a1_ptr,
                                       const float* __restrict mfa_ptr,
                                       const float* __restrict b1_ptr,
                                       float* __restrict out_ptr,
                                       int n_pixels) {

    asm volatile(
        // 设置循环计数器
        "mov x10, %[n_pixels]\n"
        "mov x11, #16\n"  // 每次处理16个元素

        // 主循环
        "1:\n"
        
        // 预取下一轮数据 (提前64字节，即16个float)
        "prfm pldl1keep, [%[a1], #128]\n"      // 预取a1数据
        "prfm pldl1keep, [%[mfa], #128]\n"     // 预取mfa数据  
        "prfm pldl1keep, [%[b1], #128]\n"      // 预取b1数据
        "prfm pstl1keep, [%[out], #128]\n"     // 预取输出缓存

        // 加载第一组数据 (16个float = 4个NEON寄存器)
        "ldp q0, q1, [%[a1]], #32\n"          // 加载a1[0:7]
        "ldp q2, q3, [%[a1]], #32\n"          // 加载a1[8:15]
        "ldp q4, q5, [%[mfa]], #32\n"         // 加载mfa[0:7]
        "ldp q6, q7, [%[mfa]], #32\n"         // 加载mfa[8:15]
        "ldp q8, q9, [%[b1]], #32\n"          // 加载b1[0:7]
        "ldp q10, q11, [%[b1]], #32\n"        // 加载b1[8:15]

        // 执行 FMA 操作: out = a1 * mfa + b1
        // 使用 fmla 指令 (Floating-point Multiply-Add)
        "fmla v8.4s, v0.4s, v4.4s\n"         // b1[0:3] += a1[0:3] * mfa[0:3]
        "fmla v9.4s, v1.4s, v5.4s\n"         // b1[4:7] += a1[4:7] * mfa[4:7]
        "fmla v10.4s, v2.4s, v6.4s\n"        // b1[8:11] += a1[8:11] * mfa[8:11]
        "fmla v11.4s, v3.4s, v7.4s\n"        // b1[12:15] += a1[12:15] * mfa[12:15]

        // 存储结果
        "stp q8, q9, [%[out]], #32\n"         // 存储结果[0:7]
        "stp q10, q11, [%[out]], #32\n"       // 存储结果[8:15]

        // 循环控制
        "subs x10, x10, x11\n"                // 减少计数器
        "b.ne 1b\n"                           // 如果不为零，跳回循环开始

        : [a1] "+r"(a1_ptr), [mfa] "+r"(mfa_ptr), [b1] "+r"(b1_ptr), [out] "+r"(out_ptr)
        : [n_pixels] "r"(n_pixels)
        : "cc", "memory", "x10", "x11", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11"
    );
}

/**
 * @brief CLAHE
 * 
 * @param in 输入图像指针8-bit
 * @param out 输出图像指针8-bit
 * @param width 图像宽度
 * @param height 图像高度
 * @param clip_limit 对比度限制阈值(例如 2.0).如果为0, 则不进行裁剪.
 */
// void clahe8u(const uint8_t* __restrict in, uint8_t* __restrict out, int width, int height, float clip_limit)
// {
//     constexpr int num_tiles_x = 4;
//     constexpr int num_tiles_y = 4;

//     if (!in || !out || width <= 0 || height <= 0 || num_tiles_x <= 0 || num_tiles_y <= 0 ||
//         width % num_tiles_x != 0 || height % num_tiles_y != 0) {
//         return;
//     }

//     constexpr int num_bins = 256;
//     const int num_tiles = num_tiles_x * num_tiles_y;
//     const int tile_w = width / num_tiles_x;
//     const int tile_h = height / num_tiles_y;
//     const int tile_pixels = tile_w * tile_h;

//     std::array<uint32_t, 4096> hist= {0};       // 4 x 4 x 256
//     std::array<uint8_t, 4096> luts_soa = {0};

//     // 1. 分块与直方图统计
//     for (int ty = 0; ty < num_tiles_y; ++ty) {
//         for (int tx = 0; tx < num_tiles_x; ++tx) {
//             const int tile_idx = ty * num_tiles_x + tx;
//             uint32_t* H = &hist[tile_idx * num_bins];
//             const int y_base = ty * tile_h;
//             const int x_base = tx * tile_w;

//             for (int y_offset = 0; y_offset < tile_h; ++y_offset) {
//                 const uint8_t* p_in_row = &in[(y_base + y_offset) * width + x_base];
//                 for (int x_offset = 0; x_offset < tile_w; ++x_offset) {
//                     ++H[p_in_row[x_offset]];
//                 }
//             }
//         }
//     }

//     // 2. 直方图裁剪与LUT计算
//     for (int i = 0; i < num_tiles; ++i) {
//         uint32_t* H = &hist[i * num_bins];

//         if (clip_limit > 0.0f) {
//             const int actual_clip_limit = std::max(1, (int)(clip_limit * tile_pixels / num_bins));

//             int clipped = 0;
//             for (int b = 0; b < num_bins; ++b) {
//                 if ((int)H[b] > actual_clip_limit) {
//                     clipped += (int)H[b] - actual_clip_limit;
//                     H[b] = (uint32_t)actual_clip_limit;
//                 }
//             }

//             const int add_all = clipped / num_bins;
//             const int residual = clipped % num_bins;
//             for (int b = 0; b < num_bins; ++b) H[b] += add_all;
//             for (int b = 0; b < residual; ++b) ++H[b];
//         }

//         uint32_t cdf = 0;
//         for (int b = 0; b < num_bins; ++b) {
//             cdf += H[b];
//             uint32_t v = (cdf * (num_bins - 1) + tile_pixels / 2) / tile_pixels;
//             luts_soa[b * num_tiles + i] = (uint8_t)std::min(v, 255u);
//         }
//     }

//     // 3. 双线性插值映射
//     constexpr int FP_SHIFT = 8;
//     constexpr int FP_ONE = 1 << FP_SHIFT;
//     const int tile_w_half = tile_w / 2;
//     const int tile_h_half = tile_h / 2;
//     const int step_fp = FP_ONE / tile_w;

//     for (int y = 0; y < height; ++y) {
//         const uint8_t* p_in_row = &in[y * width];
//         uint8_t* p_out_row = &out[y * width];

//         int ty_ref = (y - tile_h_half) / tile_h;
//         if (ty_ref < 0) ty_ref = 0;
//         int ty_top = ty_ref;
//         int ty_bottom = std::min(ty_top + 1, num_tiles_y - 1);
//         int y_center_tl = ty_top * tile_h + tile_h_half;

//         int wy_fp = ((y - y_center_tl) * FP_ONE) / tile_h;
//         if (wy_fp < 0) wy_fp = 0;
//         else if (wy_fp > FP_ONE) wy_fp = FP_ONE;

//         const int y_idx_top = ty_top * num_tiles_x;
//         const int y_idx_bottom = ty_bottom * num_tiles_x;

//         int x = 0;

//         int left_end = std::min(tile_w_half, width);
//         if (left_end > 0) {
//             const int idx_tl = y_idx_top + 0;
//             const int idx_bl = y_idx_bottom + 0;
//             for (; x < left_end; ++x) {
//                 const uint8_t pix = p_in_row[x];
//                 const uint8_t* p_lut_bin = &luts_soa[pix * num_tiles]; // luts_soa 行起始
//                 const int tl = p_lut_bin[idx_tl];
//                 const int bl = p_lut_bin[idx_bl];
//                 const int val = tl + (((bl - tl) * wy_fp) >> FP_SHIFT);
//                 p_out_row[x] = (uint8_t)val;
//             }
//         }
        
//         for (int tx_left = 0; tx_left < num_tiles_x - 1; ++tx_left) {
//             const int x_center_tl = tx_left * tile_w + tile_w_half;
//             int x_start = std::max(x, x_center_tl);
//             int x_end   = std::min(width, x_center_tl + tile_w);
//             if (x_start >= x_end) continue;

//             const int idx_tl = y_idx_top + tx_left;
//             const int idx_tr = idx_tl + 1;
//             const int idx_bl = y_idx_bottom + tx_left;
//             const int idx_br = idx_bl + 1;

//             for (int xi = x_start; xi < x_end; ++xi) {
//                 int wx_fp = ((xi - x_center_tl) * FP_ONE) / tile_w;
//                 if (wx_fp < 0) wx_fp = 0;
//                 else if (wx_fp > FP_ONE) wx_fp = FP_ONE;

//                 const uint8_t pix = p_in_row[xi];
//                 const uint8_t* p_lut_bin = &luts_soa[pix * num_tiles];

//                 const int tl = p_lut_bin[idx_tl];
//                 const int tr = p_lut_bin[idx_tr];
//                 const int bl = p_lut_bin[idx_bl];
//                 const int br = p_lut_bin[idx_br];

//                 const int top = tl + (((tr - tl) * wx_fp) >> FP_SHIFT);
//                 const int bot = bl + (((br - bl) * wx_fp) >> FP_SHIFT);
//                 const int val = top + (((bot - top) * wy_fp) >> FP_SHIFT);

//                 p_out_row[xi] = (uint8_t)val;
//             }
//             x = x_end;
//         }

//         if (x < width) {
//             const int idx_tr = y_idx_top + (num_tiles_x - 1);
//             const int idx_br = y_idx_bottom + (num_tiles_x - 1);
//             for (; x < width; ++x) {
//                 const uint8_t pix = p_in_row[x];
//                 const uint8_t* p_lut_bin = &luts_soa[pix * num_tiles];
//                 const int tr = p_lut_bin[idx_tr];
//                 const int br = p_lut_bin[idx_br];
//                 const int val = tr + (((br - tr) * wy_fp) >> FP_SHIFT);
//                 p_out_row[x] = (uint8_t)val;
//             }
//         }
//     }
// }
#define CLIP(val, min, max) ((val) < (min) ? (min) : ((val) > (max) ? (max) : (val)))

void clahe8u(const uint8_t* __restrict in, uint8_t* __restrict out, int width, int height, float clip_limit)
{
    constexpr int num_tiles_x = 4;
    constexpr int num_tiles_y = 4;
    constexpr int num_bins = 256;
    
    if (width % num_tiles_x != 0 || height % num_tiles_y != 0) return;

    const int tile_w = width / num_tiles_x;
    const int tile_h = height / num_tiles_y;
    const int tile_pixels = tile_w * tile_h;

    // 1. 内存布局优化：[Tile][Bin] (16字节对齐)
    alignas(16) uint8_t luts[num_tiles_x * num_tiles_y][num_bins];
    std::array<uint32_t, num_bins> hist_cache;

    // --- 阶段1：直方图与LUT生成 ---
    for (int ty = 0; ty < num_tiles_y; ++ty) {
        for (int tx = 0; tx < num_tiles_x; ++tx) {
            const int tile_idx = ty * num_tiles_x + tx;
            
            // 1.1 统计直方图 (手动 4 路展开)
            std::memset(hist_cache.data(), 0, sizeof(hist_cache));
            const int y_base = ty * tile_h;
            const int x_base = tx * tile_w;
            
            for (int y = 0; y < tile_h; ++y) {
                const uint8_t* ptr = in + (y_base + y) * width + x_base;
                int x = 0;
                for (; x <= tile_w - 4; x += 4) {
                    hist_cache[ptr[x]]++;
                    hist_cache[ptr[x+1]]++;
                    hist_cache[ptr[x+2]]++;
                    hist_cache[ptr[x+3]]++;
                }
                for (; x < tile_w; ++x) hist_cache[ptr[x]]++;
            }

            // 1.2 裁剪 (Clip)
            if (clip_limit > 0.0f) {
                int limit = std::max(1, (int)(clip_limit * tile_pixels / num_bins));
                int clipped = 0;
                for (int i = 0; i < num_bins; ++i) {
                    if ((int)hist_cache[i] > limit) {
                        clipped += (int)hist_cache[i] - limit;
                        hist_cache[i] = limit;
                    }
                }
                int add = clipped / num_bins;
                int rem = clipped % num_bins;
                for (int i = 0; i < num_bins; ++i) hist_cache[i] += add;
                for (int i = 0; i < rem; ++i) hist_cache[i]++;
            }

            // 1.3 生成 CDF
            uint32_t cdf = 0;
            for (int i = 0; i < num_bins; ++i) {
                cdf += hist_cache[i];
                luts[tile_idx][i] = (uint8_t)((cdf * 255 + tile_pixels / 2) / tile_pixels);
            }
        }
    }

    // --- 阶段3：双线性插值 (极致 NEON 优化) ---
    constexpr int FP_SHIFT = 8;
    constexpr int FP_ONE = 1 << FP_SHIFT;
    const int tile_w_half = tile_w / 2;
    const int tile_h_half = tile_h / 2;
    const int step_x_fp = (FP_ONE * FP_ONE) / tile_w; 

    for (int y = 0; y < height; ++y) {
        int ty_f = (y - tile_h_half);
        int ty_idx = ty_f < 0 ? 0 : ty_f / tile_h;
        int ty_rem = ty_f < 0 ? 0 : ty_f % tile_h;
        
        if (ty_idx >= num_tiles_y - 1) {
            ty_idx = num_tiles_y - 2;
            ty_rem = tile_h; 
        }
        
        const uint16_t wy = (ty_rem * FP_ONE) / tile_h;
        const uint16_t inv_wy = FP_ONE - wy;

        const int row1_idx = ty_idx * num_tiles_x;
        const int row2_idx = (ty_idx + 1) * num_tiles_x;

        const uint8_t* p_src = in + y * width;
        uint8_t* p_dst = out + y * width;

        int x = 0;

        // --- Region 1: 左边缘 ---
        if (num_tiles_x > 0) {
            const int w_limit = std::min(width, tile_w_half);
            const uint8_t* lut_tl = luts[row1_idx];
            const uint8_t* lut_bl = luts[row2_idx];
            
            for (; x <= w_limit - 16; x += 16) {
                uint8_t t_tl[16], t_bl[16];
                for(int k=0; k<16; ++k) {
                    t_tl[k] = lut_tl[p_src[x+k]];
                    t_bl[k] = lut_bl[p_src[x+k]];
                }
                uint16x8_t v_tl_lo = vmovl_u8(vld1_u8(t_tl));
                uint16x8_t v_tl_hi = vmovl_u8(vld1_u8(t_tl + 8));
                uint16x8_t v_bl_lo = vmovl_u8(vld1_u8(t_bl));
                uint16x8_t v_bl_hi = vmovl_u8(vld1_u8(t_bl + 8));

                uint16x8_t v_res_lo = vmulq_n_u16(v_tl_lo, inv_wy);
                v_res_lo = vmlaq_n_u16(v_res_lo, v_bl_lo, wy);

                uint16x8_t v_res_hi = vmulq_n_u16(v_tl_hi, inv_wy);
                v_res_hi = vmlaq_n_u16(v_res_hi, v_bl_hi, wy);

                vst1q_u8(p_dst + x, vcombine_u8(vshrn_n_u16(v_res_lo, 8), vshrn_n_u16(v_res_hi, 8)));
            }
            for (; x < w_limit; ++x) {
                uint8_t pix = p_src[x];
                uint16_t val = ((uint16_t)lut_tl[pix] * inv_wy + (uint16_t)lut_bl[pix] * wy) >> 8;
                p_dst[x] = (uint8_t)val;
            }
        }

        // --- Region 2: 中间区域 (寄存器内查表 - 黑魔法版) ---
        int32x4_t v_step_32 = vdupq_n_s32(step_x_fp);
        int32x4_t v_step_x4 = vmulq_n_s32(v_step_32, 4);
        int32x4_t v_step_x8 = vmulq_n_s32(v_step_32, 8);

        for (int tx = 0; tx < num_tiles_x - 1; ++tx) {
            int x_end = std::min(width, (tx + 1) * tile_w + tile_w_half);
            
            // 使用 base_tl 等作为基地址
            const uint8_t* base_tl = luts[row1_idx + tx];
            const uint8_t* base_tr = luts[row1_idx + tx + 1];
            const uint8_t* base_bl = luts[row2_idx + tx];
            const uint8_t* base_br = luts[row2_idx + tx + 1];

            // 1. LUT 压缩 (Downsample)
            uint8_t mini_lut[4][16];
            for(int i=0; i<16; ++i) {
                int idx = i * 16; 
                mini_lut[0][i] = base_tl[idx];
                mini_lut[1][i] = base_tr[idx];
                mini_lut[2][i] = base_bl[idx];
                mini_lut[3][i] = base_br[idx];
            }
            // 加载进寄存器，常驻！
            uint8x16_t v_lut_tl = vld1q_u8(mini_lut[0]);
            uint8x16_t v_lut_tr = vld1q_u8(mini_lut[1]);
            uint8x16_t v_lut_bl = vld1q_u8(mini_lut[2]);
            uint8x16_t v_lut_br = vld1q_u8(mini_lut[3]);

            int start_acc = (x - (tx * tile_w + tile_w_half)) * step_x_fp;
            int32x4_t v_acc_0 = vdupq_n_s32(start_acc);
            const int32_t offsets[4] = {0*step_x_fp, 1*step_x_fp, 2*step_x_fp, 3*step_x_fp};
            v_acc_0 = vaddq_s32(v_acc_0, vld1q_s32(offsets));
            int32x4_t v_acc_1 = vaddq_s32(v_acc_0, v_step_x4);

            for (; x <= x_end - 8; x += 8) {
                uint16x4_t v_wx_lo = vshrn_n_u32(vreinterpretq_u32_s32(v_acc_0), 8); 
                uint16x4_t v_wx_hi = vshrn_n_u32(vreinterpretq_u32_s32(v_acc_1), 8);
                uint16x8_t v_wx = vcombine_u16(v_wx_lo, v_wx_hi);
                uint16x8_t v_inv_wx = vsubq_u16(vdupq_n_u16(FP_ONE), v_wx);
                
                v_acc_0 = vaddq_s32(v_acc_0, v_step_x8);
                v_acc_1 = vaddq_s32(v_acc_1, v_step_x8);

                uint8x8_t v_pix = vld1_u8(p_src + x);

                // 计算索引与残差
                uint8x8_t v_idx = vshr_n_u8(v_pix, 4);
                uint8x8_t v_rem = vand_u8(v_pix, vdup_n_u8(0x0F)); 

                // Lambda: 寄存器内插值
                auto poly_interp = [&](uint8x16_t lut) -> uint16x8_t {
                    uint8x8x2_t lut_pair;
                    lut_pair.val[0] = vget_low_u8(lut);
                    lut_pair.val[1] = vget_high_u8(lut);
                    
                    uint8x8_t val0 = vtbl2_u8(lut_pair, v_idx);
                    
                    uint8x8_t v_idx_next = vadd_u8(v_idx, vdup_n_u8(1));
                    v_idx_next = vmin_u8(v_idx_next, vdup_n_u8(15));
                    uint8x8_t val1 = vtbl2_u8(lut_pair, v_idx_next);

                    // 修复：使用 vmull_u8 (u8*u8->u16) 和 vmlal_u8 (u16+u8*u8->u16)
                    // res = val0 * (16 - rem)
                    uint16x8_t res = vmull_u8(val0, vsub_u8(vdup_n_u8(16), v_rem));
                    // res += val1 * rem
                    res = vmlal_u8(res, val1, v_rem);
                    
                    return vshlq_n_u16(res, 4); // Scale to 256
                };

                uint16x8_t v_tl = poly_interp(v_lut_tl);
                uint16x8_t v_tr = poly_interp(v_lut_tr);
                uint16x8_t v_bl = poly_interp(v_lut_bl);
                uint16x8_t v_br = poly_interp(v_lut_br);

                // 将 u16 结果右移 8 位以匹配后续计算 (近似回到 u8 域)
                v_tl = vshrq_n_u16(v_tl, 8);
                v_tr = vshrq_n_u16(v_tr, 8);
                v_bl = vshrq_n_u16(v_bl, 8);
                v_br = vshrq_n_u16(v_br, 8);
                
                uint16x8_t v_top = vmulq_u16(v_tl, v_inv_wx);
                v_top = vmlaq_u16(v_top, v_tr, v_wx);
                v_top = vshrq_n_u16(v_top, 8); 

                uint16x8_t v_bot = vmulq_u16(v_bl, v_inv_wx);
                v_bot = vmlaq_u16(v_bot, v_br, v_wx);
                v_bot = vshrq_n_u16(v_bot, 8); 

                uint16x8_t v_res = vmulq_n_u16(v_top, inv_wy);
                v_res = vmlaq_n_u16(v_res, v_bot, wy);
                
                vst1_u8(p_dst + x, vshrn_n_u16(v_res, 8));
            }

            // 标量收尾部分
            for (; x < x_end; ++x) {
                int wx = (start_acc >> 8);
                // 重新计算 wx 以保证精度一致性
                wx = ((x - (tx * tile_w + tile_w_half)) * step_x_fp) >> 8;
                
                if (wx > FP_ONE) wx = FP_ONE;
                if (wx < 0) wx = 0;

                uint8_t pix = p_src[x];
                // 修复：这里使用 base_tl 而不是 lut_tl
                uint16_t top = ((uint16_t)base_tl[pix] * (FP_ONE - wx) + (uint16_t)base_tr[pix] * wx) >> 8;
                uint16_t bot = ((uint16_t)base_bl[pix] * (FP_ONE - wx) + (uint16_t)base_br[pix] * wx) >> 8;
                p_dst[x] = (uint8_t)((top * inv_wy + bot * wy) >> 8);
            }
        }

        // --- Region 3: 右边缘 ---
        if (x < width) {
            const uint8_t* lut_tr = luts[row1_idx + num_tiles_x - 1];
            const uint8_t* lut_br = luts[row2_idx + num_tiles_x - 1];
            for (; x < width; ++x) {
                 uint8_t pix = p_src[x];
                 uint16_t val = ((uint16_t)lut_tr[pix] * inv_wy + (uint16_t)lut_br[pix] * wy) >> 8;
                 p_dst[x] = (uint8_t)val;
            }
        }
    }
}

void f32_to_u16_neon(const float* __restrict src,
                            uint16_t* __restrict dst,
                            int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t a0 = vld1q_f32(src + i + 0);
        float32x4_t a1 = vld1q_f32(src + i + 4);
        // 负数快速钳到0（避免负值转无符号的未定义行为）
        const float32x4_t vzero = vdupq_n_f32(0.f);
        a0 = vmaxq_f32(a0, vzero);
        a1 = vmaxq_f32(a1, vzero);
        // 向零截断到 u32（A64: FCVTZU）
        uint32x4_t u0 = vcvtq_u32_f32(a0);
        uint32x4_t u1 = vcvtq_u32_f32(a1);
        // 窄化到 u16，饱和到 [0,65535]
        uint16x4_t h0 = vqmovn_u32(u0);
        uint16x4_t h1 = vqmovn_u32(u1);
        vst1q_u16(dst + i, vcombine_u16(h0, h1));
    }
}