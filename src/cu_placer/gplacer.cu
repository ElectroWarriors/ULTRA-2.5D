#include "gplacer.h"

__device__ const RealType GPU_LEGAL_THETA[utils::NUM_DISCRETE_VAR] = { 0.0, PI / 2.0, PI, 3.0 * PI / 2.0, 0.0, PI / 2.0, PI, 3.0 * PI / 2.0 };
__device__ const RealType GPU_LEGAL_THETA_F[utils::NUM_DISCRETE_VAR] = { 0.0, 0.0, 0.0, 0.0, PI , PI , PI , PI };

// template <typename T, int N>
// __device__ void device_gumbel_softmax_with_noise(T* z, const T* noise, T gamma) {
//     T max_z = T(0), sum = T(0);

//     for (int i = 0; i < N; ++i) { z[i] = (z[i] + noise[i]) / gamma; if (i == 0 || z[i] > max_z) max_z = z[i]; }
//     for (int i = 0; i < N; ++i) { z[i] = exp(z[i] - max_z); sum += z[i]; }
//     for (int i = 0; i < N; ++i) { z[i] /= sum; }
// }

template <typename T, int N>
__device__ void device_gumbel_softmax_with_noise(const T* z, const T* noise, T gamma, T* out) {
    T max_z = T(0), sum = T(0);

    for (int i = 0; i < N; ++i) { out[i] = (z[i] + noise[i]) / gamma; if (i == 0 || out[i] > max_z) max_z = out[i]; }
    for (int i = 0; i < N; ++i) { out[i] = exp(out[i] - max_z); sum += out[i]; }
    for (int i = 0; i < N; ++i) { out[i] /= sum; }
}

template <typename T, int N>
__device__ void device_sample_gumbel(T k, T* out) {
    unsigned int seed = threadIdx.x + blockIdx.x * blockDim.x + 1;
    for (int i = 0; i < N; ++i) {
        seed = seed * 1664525u + 1013904223u;  // LCG
        T u = ((seed & 0x00FFFFFF) + T(1e-6)) / T(16777217.0);  // ∈ (0,1)
        out[i] = -k * log(-log(u));
    }
}

__device__ void device_z2theta(gp::CUDA_Chiplet& chiplet, RealType gamma, RealType gumbel) {
    RealType z_exp[utils::NUM_DISCRETE_VAR];

    device_gumbel_softmax_with_noise<RealType, utils::NUM_DISCRETE_VAR>(chiplet.z(), chiplet.gumbel_noise(), gamma, z_exp);

    chiplet.theta() = 0;
    chiplet.theta_f() = 0;

    for (IndexType i = 0; i < utils::NUM_DISCRETE_VAR; ++i) {
        chiplet.theta() += z_exp[i] * GPU_LEGAL_THETA[i];
        chiplet.theta_f() += z_exp[i] * GPU_LEGAL_THETA_F[i];
    }

    device_sample_gumbel<RealType, utils::NUM_DISCRETE_VAR>(gumbel, chiplet.gumbel_noise());  // updates chiplet._cached_gumbel_noise
}


__global__ void updateChipletKernel(gp::CUDA_DataPackage* data_package) {
    // printf("in: %d, %d\n", blockIdx.x, threadIdx.x);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= data_package->num_chiplets) { return; }
    // printf("%d\n", idx);
    auto& chiplet = data_package->chiplets[idx];
    auto& params  = data_package->params[0];
    IndexType offset = idx * (2+utils::NUM_DISCRETE_VAR);

    chiplet.xc() = data_package->x_in[offset + 0];
    chiplet.yc() = data_package->x_in[offset + 1];
    // printf("%d, %lf, %lf\n", idx, chiplet.xc(), chiplet.yc());
    for (IndexType i=0; i<utils::NUM_DISCRETE_VAR; i++) { chiplet.z(i) = data_package->x_in[offset + 2 + i]; }

    device_z2theta(chiplet, params.curr_at(0, 0), params.curr_at(0, 3));
}

__global__ void computeHpwlKernel(gp::CUDA_DataPackage* data_package) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= data_package->num_hpwldatas) { return; }
    auto& chiplets = data_package->chiplets;
    auto& d = data_package->hpwldatas[idx];
    for (int i = 0; i < 2+utils::NUM_DISCRETE_VAR; ++i) { d.grad0[i] = RealType(0); d.grad1[i] = RealType(0); }

    IndexType c0_idx = d.c0_idx, c1_idx = d.c1_idx, param_row = d.param_row;
    gp::CUDA_Param& param = data_package->params[0];
    RealType& gamma = param.curr_at(param_row, 0);
    RealType& x0 = chiplets[c0_idx].xc();
    RealType& y0 = chiplets[c0_idx].yc();
    RealType& x1 = chiplets[c1_idx].xc();
    RealType& y1 = chiplets[c1_idx].yc();
    RealType& theta0 = chiplets[c0_idx].theta();
    RealType& theta1 = chiplets[c1_idx].theta();
    RealType& theta_f0 = chiplets[c0_idx].theta_f();
    RealType& theta_f1 = chiplets[c1_idx].theta_f();
    RealType (&z0)[utils::NUM_DISCRETE_VAR] = chiplets[c0_idx].z();
    RealType (&z1)[utils::NUM_DISCRETE_VAR] = chiplets[c0_idx].z();
    RealType (&gumbel0)[utils::NUM_DISCRETE_VAR] = chiplets[c0_idx].gumbel_noise();
    RealType (&gumbel1)[utils::NUM_DISCRETE_VAR] = chiplets[c0_idx].gumbel_noise();

    RealType cos_theta0  = cos(theta0),   sin_theta0  = sin(theta0);
    RealType cos_theta1  = cos(theta1),   sin_theta1  = sin(theta1);
    RealType cos_thetaf0 = cos(theta_f0), sin_thetaf0 = sin(theta_f0);
    RealType cos_thetaf1 = cos(theta_f1), sin_thetaf1 = sin(theta_f1);

    RealType cp0_x = x0 + cos_theta0 * (d.off_x0*cos_thetaf0) - d.off_y0 * sin_theta0;
    RealType cp0_y = y0 + cos_theta0 * d.off_y0 + (d.off_x0*cos_thetaf0) * sin_theta0;
    RealType cp1_x = x1 + cos_theta1 * (d.off_x1*cos_thetaf1) - d.off_y1 * sin_theta1;
    RealType cp1_y = y1 + cos_theta1 * d.off_y1 + (d.off_x1*cos_thetaf1) * sin_theta1;

    RealType cp_dx = cp0_x - cp1_x;
    RealType cp_dy = cp0_y - cp1_y;

    d.value = fabs(cp_dx) + fabs(cp_dy);
    d.grad0[0] = 2 * param.curr_at(param_row, 1) * cp_dx, d.grad1[0] = -d.grad0[0];
    d.grad0[1] = 2 * param.curr_at(param_row, 1) * cp_dy, d.grad1[1] = -d.grad0[1];

    RealType pW_px0 = 2 * cp_dx, pW_py0 = 2 * cp_dy, pW_pz0 = 2 * d.off_x0 * sin_thetaf0;
    RealType pW_px1 = -pW_px0  , pW_py1 = -pW_py0  , pW_pz1 = 2 * d.off_x1 * sin_thetaf1;

    RealType px0_pthetaf0 = - d.off_x0 * cos_theta0 * sin_thetaf0;
    RealType py0_pthetaf0 = - d.off_x0 * sin_theta0 * sin_thetaf0;
    RealType pz0_pthetaf0 =   d.off_x0 * cos_thetaf0;
    RealType px1_pthetaf1 = - d.off_x1 * cos_theta1 * sin_thetaf1;
    RealType py1_pthetaf1 = - d.off_x1 * sin_theta1 * sin_thetaf1;
    RealType pz1_pthetaf1 =   d.off_x1 * cos_thetaf1;

    RealType pW_pthetaf0 = pW_px0 * px0_pthetaf0 + pW_py0 * py0_pthetaf0 + pW_pz0 * pz0_pthetaf0;
    RealType pW_pthetaf1 = pW_px1 * px1_pthetaf1 + pW_py1 * py1_pthetaf1 + pW_pz1 * pz1_pthetaf1;

    RealType px0_ptheta0 = -(sin_theta0*cos_thetaf0) * d.off_x0 - cos_theta0 * d.off_y0;
    RealType py0_ptheta0 =  (cos_theta0*cos_thetaf0) * d.off_x0 - sin_theta0 * d.off_y0;
    RealType px1_ptheta1 = -(sin_theta1*cos_thetaf1) * d.off_x1 - cos_theta1 * d.off_y1;
    RealType py1_ptheta1 =  (cos_theta1*cos_thetaf1) * d.off_x1 - sin_theta1 * d.off_y1;

    RealType pW_ptheta0 = pW_px0 * px0_ptheta0 + pW_py0 * py0_ptheta0;
    RealType pW_ptheta1 = pW_px1 * px1_ptheta1 + pW_py1 * py1_ptheta1;

    RealType z0_exp[utils::NUM_DISCRETE_VAR];
    RealType z1_exp[utils::NUM_DISCRETE_VAR];
    device_gumbel_softmax_with_noise<RealType, utils::NUM_DISCRETE_VAR>(z0, gumbel0, gamma, z0_exp);
    device_gumbel_softmax_with_noise<RealType, utils::NUM_DISCRETE_VAR>(z1, gumbel1, gamma, z1_exp);

    for (IndexType i=0; i<utils::NUM_DISCRETE_VAR; i++) { 
        // if (idx == 1) printf("hpwl check: %lf, %lf, %lf, %lf\n", param.curr_at(param_row, 2), pW_ptheta0, z0_exp[i], theta0);
        d.grad0[i+2] += param.curr_at(param_row, 2) * pW_ptheta0  * (z0_exp[i] / gamma) * (GPU_LEGAL_THETA[i]   - theta0 ); 
        d.grad1[i+2] += param.curr_at(param_row, 2) * pW_ptheta1  * (z1_exp[i] / gamma) * (GPU_LEGAL_THETA[i]   - theta1 ); 
        d.grad0[i+2] += param.curr_at(param_row, 4) * pW_pthetaf0 * (z0_exp[i] / gamma) * (GPU_LEGAL_THETA_F[i] - theta_f0); 
        d.grad1[i+2] += param.curr_at(param_row, 4) * pW_pthetaf1 * (z1_exp[i] / gamma) * (GPU_LEGAL_THETA_F[i] - theta_f1); 
    }

}

__global__ void computeBdyKernel(gp::CUDA_DataPackage* data_package) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= data_package->num_bdydatas) { return; }
    auto& chiplets = data_package->chiplets;
    auto& d = data_package->bdydatas[idx];
    d.grad[0] = RealType(0);
    d.grad[1] = RealType(0);

    IndexType c_idx = d.c_idx, param_row = d.param_row;
    gp::CUDA_Param& param = data_package->params[0];
    RealType& gamma = param.curr_at(param_row, 0);
    RealType& xc = chiplets[c_idx].xc();
    RealType& yc = chiplets[c_idx].yc();
    RealType& wc = chiplets[c_idx].w();
    RealType& hc = chiplets[c_idx].h();

    RealType& theta = chiplets[c_idx].theta();
    RealType cos_2theta = RealType(0.5) * (RealType(1.0) + cos(2.0 * theta));
    RealType w = cos_2theta * wc + (RealType(1.0) - cos_2theta) * hc;
    RealType h = cos_2theta * hc + (RealType(1.0) - cos_2theta) * wc;

    RealType area_c = w * h;

    RealType c_xl = xc - w / 2;
    RealType c_xh = xc + w / 2;
    RealType c_yl = yc - h / 2;
    RealType c_yh = yc + h / 2;

    RealType b_xl = d.xr - d.wr / 2;
    RealType b_xh = d.xr + d.wr / 2;
    RealType b_yl = d.yr - d.hr / 2;
    RealType b_yh = d.yr + d.hr / 2;

    // === overlapXY<Point>::absmul() logic ===
    RealType ox = max(c_xh, b_xh) - min(c_xl, b_xl) - (w + d.wr);
    RealType oy = max(c_yh, b_yh) - min(c_yl, b_yl) - (h + d.hr);
    RealType absmul = fabs(ox) * fabs(oy);

    d.value = area_c - absmul; 

    // === Gradient computation ===
    RealType cos_theta = cos(theta);
    RealType sin_theta = sin(theta);
    RealType half_w = wc * RealType(0.5);
    RealType half_h = hc * RealType(0.5);

    // Rotated corners
    RealType corner_x[4] = { half_w, -half_w, -half_w,  half_w };
    RealType corner_y[4] = { half_h,  half_h, -half_h, -half_h };

    RealType x_min = LOC_TYPE_MAX, y_min = LOC_TYPE_MAX;
    RealType x_max = LOC_TYPE_MIN, y_max = LOC_TYPE_MIN;

    for (int k = 0; k < 4; ++k) {
        RealType x_rot = xc + corner_x[k] * cos_theta - corner_y[k] * sin_theta;
        RealType y_rot = yc + corner_x[k] * sin_theta + corner_y[k] * cos_theta;

        x_min = min(x_min, x_rot);
        y_min = min(y_min, y_rot);
        x_max = max(x_max, x_rot);
        y_max = max(y_max, y_rot);
    }

    // Compare against boundary box
    RealType gx = RealType(0), gy = RealType(0);
    gx += min(x_min - b_xl, RealType(0));
    gy += min(y_min - b_yl, RealType(0));
    gx += max(x_max - b_xh, RealType(0));
    gy += max(y_max - b_yh, RealType(0));

    d.grad[0] = param.curr_at(param_row, 1) * gx * RealType(100.0) * fabs(gx);
    d.grad[1] = param.curr_at(param_row, 1) * gy * RealType(100.0) * fabs(gy);
}

__global__ void computeOvlKernel(gp::CUDA_DataPackage* data_package) {
    IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= data_package->num_ovldatas) { return; }

    auto& chiplets = data_package->chiplets;
    auto& d = data_package->ovldatas[idx];
    for (int i = 0; i < 2; i++) { d.grad0[i] = RealType(0); d.grad1[i] = RealType(0); }

    IndexType c0_idx = d.c0_idx, c1_idx = d.c1_idx, param_row = d.param_row;
    gp::CUDA_Param& param = data_package->params[0];
    RealType& gamma = param.curr_at(param_row, 0);
    RealType& x0 = chiplets[c0_idx].xc();
    RealType& y0 = chiplets[c0_idx].yc();
    RealType& x1 = chiplets[c1_idx].xc();
    RealType& y1 = chiplets[c1_idx].yc();

    RealType& theta0 = chiplets[c0_idx].theta();
    RealType& theta1 = chiplets[c1_idx].theta();

    RealType cos2_0 = RealType(0.5) * (RealType(1.0) + cos(2.0 * theta0));
    RealType cos2_1 = RealType(0.5) * (RealType(1.0) + cos(2.0 * theta1));

    RealType w0 = cos2_0 * chiplets[c0_idx].w() + (RealType(1.0) - cos2_0) * chiplets[c0_idx].h();
    RealType h0 = cos2_0 * chiplets[c0_idx].h() + (RealType(1.0) - cos2_0) * chiplets[c0_idx].w();
    RealType w1 = cos2_1 * chiplets[c1_idx].w() + (RealType(1.0) - cos2_1) * chiplets[c1_idx].h();
    RealType h1 = cos2_1 * chiplets[c1_idx].h() + (RealType(1.0) - cos2_1) * chiplets[c1_idx].w();

    // === Value computation ===
    RealType dx0 = x0 - x1 + w0 / 2 + w1 / 2;
    RealType dx1 = x1 - x0 + w0 / 2 + w1 / 2;
    RealType dy0 = y0 - y1 + h0 / 2 + h1 / 2;
    RealType dy1 = y1 - y0 + h0 / 2 + h1 / 2;

    RealType exa = exp(-dx0 / gamma);
    RealType exb = exp(-dx1 / gamma);
    RealType eya = exp(-dy0 / gamma);
    RealType eyb = exp(-dy1 / gamma);

    RealType ovlx = gamma * log(RealType(1.0) / (exa + exb) + RealType(1.0));
    RealType ovly = gamma * log(RealType(1.0) / (eya + eyb) + RealType(1.0));

    d.value = ovlx * ovly;

    // === Gradient computation ===
    RealType denom_x = (RealType(1.0) / (exa + exb) + RealType(1.0)) * (exa + exb) * (exa + exb);
    RealType denom_y = (RealType(1.0) / (eya + eyb) + RealType(1.0)) * (eya + eyb) * (eya + eyb);

    RealType dx0_grad = -gamma * ovly * (-exa + exb) / denom_x;
    RealType dy0_grad = -gamma * ovlx * (-eya + eyb) / denom_y;

    d.grad0[0] =  param.curr_at(param_row, 1) * dx0_grad;
    d.grad0[1] =  param.curr_at(param_row, 1) * dy0_grad;
    d.grad1[0] = -param.curr_at(param_row, 1) * dx0_grad;
    d.grad1[1] = -param.curr_at(param_row, 1) * dy0_grad;
}

__global__ void computeCraKernel(gp::CUDA_DataPackage* data_package) {
    IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= data_package->num_cradatas) { return; }

    auto& chiplets = data_package->chiplets;
    auto& d = data_package->cradatas[idx];
    for (int i = 0; i < 2; ++i) { d.grad[i] = RealType(0); }
    IndexType c_idx = d.c_idx, param_row = d.param_row;
    gp::CUDA_Param& param = data_package->params[0];
    RealType& gamma = param.curr_at(param_row, 0);
    RealType& theta = chiplets[c_idx].theta();
    RealType cos_2theta = RealType(0.5) * (RealType(1.0) + cos(2.0 * theta));

    RealType& xc = chiplets[c_idx].xc();
    RealType& yc = chiplets[c_idx].yc();
    RealType& w  = chiplets[c_idx].w();
    RealType& h  = chiplets[c_idx].h();
    RealType wc = cos_2theta * w + (RealType(1.0) - cos_2theta) * h;
    RealType hc = cos_2theta * h + (RealType(1.0) - cos_2theta) * w;
    RealType& xr = d.xr, yr = d.yr;
    RealType& wr = d.wr, hr = d.hr;

    // === Value computation ===
    RealType dx0 = xc - xr + wc / 2 + wr / 2;
    RealType dx1 = xr - xc + wc / 2 + wr / 2;
    RealType dy0 = yc - yr + hc / 2 + hr / 2;
    RealType dy1 = yr - yc + hc / 2 + hr / 2;

    RealType exa = exp(-dx0 / gamma);
    RealType exb = exp(-dx1 / gamma);
    RealType eya = exp(-dy0 / gamma);
    RealType eyb = exp(-dy1 / gamma);

    RealType ovlx = gamma * log(RealType(1.0) / (exa + exb) + RealType(1.0));
    RealType ovly = gamma * log(RealType(1.0) / (eya + eyb) + RealType(1.0));
    RealType ovl = ovlx * ovly;

    RealType eovl_term = (ovl - d.Amin) / gamma;
    if (eovl_term >= RealType(10.0)) {
        d.value = RealType(2.0) * d.Amin;
        d.grad[0] = RealType(0.0);
        d.grad[1] = RealType(0.0);
        return;
    }

    RealType eovl = exp(eovl_term);
    d.value = RealType(2.0) * d.Amin - ovl + gamma * log(RealType(1.0) + eovl);

    // === Gradient computation ===
    RealType pA_pxc = -ovly * (-exa + exb) / ((RealType(1.0) + exa + exb) * (exa + exb));
    RealType pA_pyc = -ovlx * (-eya + eyb) / ((RealType(1.0) + eya + eyb) * (eya + eyb));

    RealType norm = sqrt(pA_pxc * pA_pxc + pA_pyc * pA_pyc);
    if (norm > RealType(0.0)) {
        pA_pxc /= norm;
        pA_pyc /= norm;
    }

    RealType k = (d.Amin > ovl) ? sqrt(d.Amin - ovl) : RealType(1.0);
    RealType dxc = k * (-RealType(1.0) / (eovl + RealType(1.0))) * pA_pxc;
    RealType dyc = k * (-RealType(1.0) / (eovl + RealType(1.0))) * pA_pyc;

    d.grad[0] = param.curr_at(param_row, 1) * dxc;
    d.grad[1] = param.curr_at(param_row, 1) * dyc;
}

__global__ void clearGradientKernel(gp::CUDA_DataPackage* data_package, IndexType N) {
    for (IndexType i = 0; i<N; i++) { data_package->grad[i] = 0; }
}

__global__ void accumulateHpwlGradKernel(gp::CUDA_DataPackage* data_package) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= data_package->num_hpwldatas) { return; }
    // auto& chiplets = data_package->chiplets;
    auto& d = data_package->hpwldatas[idx];

    IndexType circ = 2+utils::NUM_DISCRETE_VAR;
    IndexType c0_idx = d.c0_idx, c1_idx = d.c1_idx;
    for (int i = 0; i < circ; ++i) {
        atomicAdd(&data_package->grad[c0_idx*circ+i], d.grad0[i]);
        atomicAdd(&data_package->grad[c1_idx*circ+i], d.grad1[i]);
    }
}

__global__ void accumulateBdyGradKernel(gp::CUDA_DataPackage* data_package) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= data_package->num_bdydatas) { return; }

    // auto& chiplets = data_package->chiplets;
    auto& d = data_package->bdydatas[idx];

    IndexType circ = 2+utils::NUM_DISCRETE_VAR;
    IndexType c_idx = d.c_idx;
    for (int i = 0; i < 2; ++i) { atomicAdd(&data_package->grad[c_idx*circ+i], d.grad[i]); }
}

__global__ void accumulateOvlGradKernel(gp::CUDA_DataPackage* data_package) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= data_package->num_ovldatas) { return; }

    // auto& chiplets = data_package->chiplets;
    auto& d = data_package->ovldatas[idx];

    IndexType circ = 2+utils::NUM_DISCRETE_VAR;
    IndexType c0_idx = d.c0_idx, c1_idx = d.c1_idx;
    for (int i = 0; i < 2; ++i) { 
        atomicAdd(&data_package->grad[c0_idx*circ+i], d.grad0[i]); atomicAdd(&data_package->grad[c1_idx*circ+i], d.grad1[i]);
    }
}

__global__ void accumulateCraGradKernel(gp::CUDA_DataPackage* data_package) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= data_package->num_cradatas) { return; }

    // auto& chiplets = data_package->chiplets;
    auto& d = data_package->cradatas[idx];

    IndexType circ = 2+utils::NUM_DISCRETE_VAR;
    IndexType c_idx = d.c_idx;
    for (int i = 0; i < 2; ++i) { atomicAdd(&data_package->grad[c_idx*circ+i], d.grad[i]); }
}

__global__ void accumulateValueKernel(gp::CUDA_DataPackage* data_package) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 1) { return; }

    // auto& chiplets = data_package->chiplets;
    // auto& d = data_package->cradatas[idx];
    auto& value = data_package->value;

    value = 0;
    for (IndexType i=0; i<data_package->num_hpwldatas; i++) { value += data_package->hpwldatas[i].value; }
    for (IndexType i=0; i<data_package->num_bdydatas ; i++) { value += data_package->bdydatas[i].value; }
    for (IndexType i=0; i<data_package->num_ovldatas; i++) { value += data_package->ovldatas[i].value; }
    for (IndexType i=0; i<data_package->num_cradatas; i++) { value += data_package->cradatas[i].value; }
}

__global__ void updateParameterKernel(gp::CUDA_DataPackage* data_package) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 1) { return; }

    auto& params = data_package->params;
    auto& updater = data_package->param_updaters;
    for (IndexType i=0; i<params->size(); i++) {
        if (updater->enable_at(i)) {
            params->curr_at(i) = updater->target_at(i) + (params->curr_at(i) - updater->target_at(i)) * updater->mul_at(i);
        }
    }
}

__global__ void printChipletMsgKernel(gp::CUDA_DataPackage* data_package) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 1) { return; }

    auto& chiplets = data_package->chiplets;
    for (IndexType i=0; i<data_package->num_chiplets; i++) { chiplets[i].print_info(); }
}


namespace gp {

void launchChipletUpdateKernel(CUDA_DataPackage* data_package, IndexType N) {
    int block = 256, grid = (N + block - 1) / block;
    updateChipletKernel<<<grid, block>>>(data_package);
}

void launchHpwlKernel(CUDA_DataPackage* data_package, IndexType N) {
    int block = 256, grid = (N + block - 1) / block;
    computeHpwlKernel<<<grid, block>>>(data_package);
}

void launchBdyKernel(CUDA_DataPackage* data_package, IndexType N) {
    int block = 256, grid = (N + block - 1) / block;
    computeBdyKernel<<<grid, block>>>(data_package);
}

void launchOvlKernel(CUDA_DataPackage* data_package, IndexType N) {
    int block = 256, grid = (N + block - 1) / block;
    computeOvlKernel<<<grid, block>>>(data_package);
}

void launchCraKernel(CUDA_DataPackage* data_package, IndexType N) {
    int block = 256, grid = (N + block - 1) / block;
    computeCraKernel<<<grid, block>>>(data_package);
}

void launchGradientClearKernel(CUDA_DataPackage* data_package, IndexType N) {
    clearGradientKernel<<<1, 1>>>(data_package, N);
}

void launchHpwlGradAccumulateKernel(CUDA_DataPackage* data_package, IndexType N) {
    int block = 256, grid = (N + block - 1) / block;
    accumulateHpwlGradKernel<<<grid, block>>>(data_package);
}

void launchBdyGradAccumulateKernel(CUDA_DataPackage* data_package, IndexType N) {
    int block = 256, grid = (N + block - 1) / block;
    accumulateBdyGradKernel<<<grid, block>>>(data_package);
}

void launchOvlGradAccumulateKernel(CUDA_DataPackage* data_package, IndexType N) {
    int block = 256, grid = (N + block - 1) / block;
    accumulateOvlGradKernel<<<grid, block>>>(data_package);
}

void launchCraGradAccumulateKernel(CUDA_DataPackage* data_package, IndexType N) {
    int block = 256, grid = (N + block - 1) / block;
    accumulateCraGradKernel<<<grid, block>>>(data_package);
}

void launchValueAccumulateKernel(CUDA_DataPackage* data_package) {
    accumulateValueKernel<<<1, 1>>>(data_package);
}

void launchParamUpdateKernel(CUDA_DataPackage* data_package) {
    updateParameterKernel<<<1, 1>>>(data_package);
}

void launchChipletPrintKernel(CUDA_DataPackage* data_package) {
    printChipletMsgKernel<<<1, 1>>>(data_package);
}






}



