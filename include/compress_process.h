#pragma once
#include <cstdint>
/*
 * 压缩主流程：raw1 (16bit 输入) -> img_8bit (8bit 输出)
 * 参数:
 *  raw1            输入原始 14bit 数据 (height*width)
 *  raw11, raw2     中间缓冲 (由调用方分配)
 *  Hist_all...     直方图相关缓冲
 *  rearrange_table 重排表缓冲
 *  img_8bit        输出 8bit 图像缓冲
 *  oriscale        输入灰度最大值, 默认2047
 *  centerTarget    输出分段压缩中心点目标（如 80）
 *  outscale_max outscale_min   输出最大\小值（如 235）
 *  ratio_low, ratio_high 低高区间压缩比
 *  img8_min        输出 8bit 图像最小值
 */

void compress_process(
    const uint16_t* raw1,
    uint16_t* raw11,
    uint16_t* raw2,
    uint32_t* Hist_all,
    uint32_t* Hist_filtered1,
    uint32_t* Hist_filtered2,
    uint16_t* rearrange_table,
    uint8_t*  img_8bit,
    int height,
    int width,
    uint16_t oriscale,
    int centerTarget,
    int outscale_min,
    int outscale_max,
    float& ratio_low,
    float& ratio_high,
    uint8_t& img8_min
);

