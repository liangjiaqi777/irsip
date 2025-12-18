#include <vector>
#include <iostream>

void nuc_apply_neon_scalar_gain(const uint16_t* __restrict scene16, const uint16_t* __restrict shutter16,
                                float gain, float* __restrict image_out, size_t n);
void mfa_fuse_once_neon(const float* __restrict prev_frame, const float* __restrict cur_frame,
                            float* __restrict out_frame, size_t n);
void guidedfilter_coeffs_two_eps(
    const std::vector<float>& p, int width, int height,
    float eps1, float eps2,
    std::vector<float>& a1, std::vector<float>& b1,
    std::vector<float>& a2, std::vector<float>& b2,
    std::vector<float>& workspace);
void guided_filter_fma_neon_asm(const float* __restrict a1_ptr,
                                const float* __restrict mfa_ptr,
                                const float* __restrict b1_ptr,
                                float* __restrict out_ptr,
                                int n_pixels);
void clahe8u(const uint8_t* __restrict in, uint8_t* __restrict out, int width, int height, float clip_limit);
std::vector<float> guidedfilter_self(const std::vector<float>& p, 
    int width, int height, float eps, std::vector<float>& workspace);
void f32_to_u16_neon(const float* __restrict src,
                            uint16_t* __restrict dst,
                            int n);
void guided_filter_compute_edge_neon(const float* gfp, 
                                     const float* a2p, 
                                     const float* mfp, 
                                     const float* b2p, 
                                     float* edgep, 
                                     int n_pixels);