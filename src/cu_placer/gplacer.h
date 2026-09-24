

#ifndef GPLACER_H_
#define GPLACER_H_

#include<iostream>
#include<string>
#include <random>
#include <cmath>
#include <array>
#include <algorithm>
#include <cuda_runtime.h>

#include"utils.h"
#include"db.h"
#include"gpu_alloc.h"
#include"armadillo"

using namespace db;

namespace gp {

constexpr IndexType NUM_CONS = 4;
constexpr IndexType NUM_HPWL_PARAM = 5;
constexpr IndexType NUM_BDY_PARAM = 2;
constexpr IndexType NUM_OVL_PARAM = 2;
constexpr IndexType NUM_CRA_PARAM = 2;
constexpr IndexType NUM_MAX_PARAM = std::max(std::max(NUM_HPWL_PARAM, NUM_BDY_PARAM), std::max(NUM_OVL_PARAM, NUM_CRA_PARAM));

using HpwlInput = std::tuple<ClusterType*, ClusterType*>;
using HpwlParams =std::array<RealType, NUM_HPWL_PARAM>; // [gamma, w_xy, w_alpha, gumbel, w_alphaf]
using HpwlGrad =std::array<RealType, 2+utils::NUM_DISCRETE_VAR>; // [x, y, z0, z1, z2, z3]

using BdyInput = std::tuple<Chiplet*, Point, Size>;
using BdyParams =std::array<RealType, NUM_BDY_PARAM>; // [gamma, w_bdy]
using BdyGrad =std::array<RealType, 2>; // [x, y]

using OvlInput = std::tuple<Chiplet*, Chiplet*>;
using OvlParams =std::array<RealType, NUM_OVL_PARAM>; // [gamma, w_ovl]
using OvlGrad =std::array<RealType, 2>; // [x, y]

using CraInput = std::tuple<Chiplet*, Reticle*, LocType>;
using CraParams =std::array<RealType, NUM_CRA_PARAM>; // [gamma, w_cra]
using CraGrad =std::array<RealType, 2>; // x, y

// =============================================================================== //
//                          cuda gpu data structure                                //
// =============================================================================== //
struct CUDA_Chiplet {
    IndexType _id;
    RealType _loc[2];
    RealType _size[2];
    RealType _theda;
    RealType _theda_f;
    RealType _z[utils::NUM_DISCRETE_VAR];
    RealType _cached_gumbel_noise[utils::NUM_DISCRETE_VAR];
    RealType _grad[2+utils::NUM_DISCRETE_VAR];

    __HOST__ __DEVICE__ CUDA_Chiplet() {}
    __HOST__            CUDA_Chiplet(Chiplet* chiplet) {
        _id = chiplet->id();
        _loc[0] = chiplet->xc();
        _loc[1] = chiplet->yc();
        _size[0] = chiplet->w();
        _size[1] = chiplet->h();
        _theda = chiplet->theta();
        _theda_f = chiplet->theta_f();
        // z, gumbel etc.
        for (IndexType i = 0; i < utils::NUM_DISCRETE_VAR; ++i) {
            _z[i] = chiplet->z()[i];
            _cached_gumbel_noise[i] = chiplet->gumbel_noise()[i];
        }
    }

    __HOST__ __DEVICE__ IndexType& id() { return _id; }
    __HOST__ __DEVICE__ RealType& xc() { return _loc[0]; }
    __HOST__ __DEVICE__ RealType& yc() { return _loc[1]; }
    __HOST__ __DEVICE__ RealType& w() { return _size[0]; }
    __HOST__ __DEVICE__ RealType& h() { return _size[1]; }
    __HOST__ __DEVICE__ RealType& theta() { return _theda; }
    __HOST__ __DEVICE__ RealType& theta_f() { return _theda_f; }
    __HOST__ __DEVICE__ RealType& z(IndexType index) { return _z[index]; }
    __HOST__ __DEVICE__ RealType (&z())[utils::NUM_DISCRETE_VAR] { return _z; }
    __HOST__ __DEVICE__ RealType (&gumbel_noise())[utils::NUM_DISCRETE_VAR] { return _cached_gumbel_noise; }
    __HOST__ __DEVICE__ RealType (&grad())[2 + utils::NUM_DISCRETE_VAR] { return _grad; }

    __HOST__ __DEVICE__ void clear_grad() { for (IndexType i=0; i<2+utils::NUM_DISCRETE_VAR; i++) { _grad[i] = RealType(0); } }
    template <std::size_t N>
    __HOST__ __DEVICE__ void acc_grad(const RealType (&g)[N]) { for (IndexType i=0; i<N; i++) { _grad[i] += g[i]; } }

    __DEVICE__ void print_info() const { printf("%d %.3f %.3f %.3f %.3f %.3f %.3f\n", _id, _loc[0], _loc[1], _theda*180/3.14, _theda_f*180/3.14, _size[0], _size[1]); }
};

struct CUDA_HpwlData {
    IndexType c0_idx, c1_idx;
    LocType off_x0, off_y0, off_x1, off_y1;
    IndexType param_row;
    RealType grad0[2+utils::NUM_DISCRETE_VAR];
    RealType grad1[2+utils::NUM_DISCRETE_VAR];
    RealType value;
};

struct CUDA_BdyData {
    IndexType c_idx;
    LocType xr, yr, wr, hr;
    IndexType param_row;
    RealType grad[2];
    RealType value;
};

struct CUDA_OvlData {
    IndexType c0_idx, c1_idx;
    IndexType param_row;
    RealType grad0[2];
    RealType grad1[2];
    RealType value;
};

struct CUDA_CraData {
    IndexType c_idx;
    LocType xr, yr, wr, hr;
    RealType Amin;
    IndexType param_row;
    RealType grad[2];
    RealType value;
};

struct CUDA_Param {
    static constexpr IndexType n_rows = NUM_CONS;
    static constexpr IndexType n_cols = NUM_MAX_PARAM;
    RealType param[n_rows * n_cols];

    __HOST__ __DEVICE__ inline IndexType size() const { return n_rows * n_cols; }
    __DEVICE__ inline RealType& curr_at(IndexType i)   { return param[i]; }
    __HOST__ __DEVICE__ inline RealType& curr_at(IndexType r, IndexType c)   { return param[r * n_cols + c]; }
    __HOST__ void setparams(HpwlParams p0, BdyParams p1, OvlParams p2, CraParams p3) { 
        for (IndexType c = 0; c < n_cols; c++) { 
            curr_at(0, c) = p0[c]; curr_at(1, c) = p1[c]; curr_at(2, c) = p2[c]; curr_at(3, c) = p3[c]; 
        } 
    }
};


struct CUDA_ParamUpdater {
    static constexpr IndexType n_rows = NUM_CONS;
    static constexpr IndexType n_cols = NUM_MAX_PARAM;
    bool     enable[n_rows * n_cols];
    RealType param[n_rows * n_cols];
    RealType target[n_rows * n_cols];
    RealType mul[n_rows * n_cols];

    __HOST__ CUDA_ParamUpdater() {
        for (IndexType i = 0; i < size(); ++i) {
            enable[i] = false; target[i] = RealType(0); mul[i] = RealType(1);
        }
    }

    __HOST__ void all_disable() { for (IndexType i = 0; i < size(); ++i) {enable[i] = false; } }

    __HOST__ __DEVICE__ inline IndexType size() const { return n_rows * n_cols; }
    __DEVICE__ inline bool&     enable_at(IndexType i) { return enable[i]; }
    __DEVICE__ inline RealType& target_at(IndexType i) { return target[i]; }
    __DEVICE__ inline RealType& mul_at(IndexType i)    { return mul[i]; }

    __HOST__ __DEVICE__ inline bool&     enable_at(IndexType r, IndexType c) { return enable[r * n_cols + c]; }
    __HOST__ __DEVICE__ inline RealType& target_at(IndexType r, IndexType c) { return target[r * n_cols + c]; }
    __HOST__ __DEVICE__ inline RealType& mul_at(IndexType r, IndexType c)    { return mul[r * n_cols + c]; }
};


struct CUDA_DataPackage {
    IndexType num_chiplets;
    IndexType num_hpwldatas;
    IndexType num_bdydatas;
    IndexType num_ovldatas;
    IndexType num_cradatas;
    IndexType num_iodata;
    const IndexType num_params = 1;
    const IndexType num_param_updaters = 1;
    CUDA_Chiplet* chiplets;
    CUDA_HpwlData* hpwldatas;
    CUDA_BdyData*  bdydatas;
    CUDA_OvlData*  ovldatas;
    CUDA_CraData*  cradatas;
    CUDA_Param*    params;
    CUDA_ParamUpdater* param_updaters;
    // input (x) and output (grad and value)
    RealType* x_in;
    RealType  value;
    RealType* grad;
};

// call kernel
void launchChipletUpdateKernel(CUDA_DataPackage* data_package, IndexType N);
void launchHpwlKernel(CUDA_DataPackage* data_package, IndexType N);
void launchBdyKernel(CUDA_DataPackage* data_package, IndexType N);
void launchOvlKernel(CUDA_DataPackage* data_package, IndexType N);
void launchCraKernel(CUDA_DataPackage* data_package, IndexType N);

void launchGradientClearKernel(CUDA_DataPackage* data_package, IndexType N);

void launchHpwlGradAccumulateKernel(CUDA_DataPackage* data_package, IndexType N);
void launchBdyGradAccumulateKernel(CUDA_DataPackage* data_package, IndexType N);
void launchOvlGradAccumulateKernel(CUDA_DataPackage* data_package, IndexType N);
void launchCraGradAccumulateKernel(CUDA_DataPackage* data_package, IndexType N);

void launchValueAccumulateKernel(CUDA_DataPackage* data_package);

void launchParamUpdateKernel(CUDA_DataPackage* data_package);

void launchChipletPrintKernel(CUDA_DataPackage* data_package);
// =============================================================================== //
//                                cpu calculator                                   //
// =============================================================================== //
template<typename Derived, typename InputTuple, typename InputParams>
class Calculator {
public:
    explicit Calculator(const InputTuple& input, const InputParams& param) : _input(input), _params(param) {}
    void evaluate() { static_cast<Derived*>(this)->evaluate_impl(); }
    void gradient() { static_cast<Derived*>(this)->gradient_impl(); }
    void accumulate_gradient() { static_cast<Derived*>(this)->accumulate_gradient_impl(); }
    RealType value() { return static_cast<Derived*>(this)->value(); }

protected:
    InputTuple _input;
    const InputParams& _params;
    const InputTuple& input_tuple() const { return _input; }
    template <IndexType Index>
    decltype(auto) get_input() const { return std::get<Index>(_input); }
};


class HpwlCalculator : public Calculator<HpwlCalculator, HpwlInput, HpwlParams> {
public:
    explicit HpwlCalculator(const HpwlInput& input, const HpwlParams& param) : Calculator(input, param) {
        auto [cl0, cl1] = input; 
        _cl0 = cl0; _c0 = _cl0->chiplet(); _cp0_off = _cl0->offset_center();
        _cl1 = cl1; _c1 = _cl1->chiplet(); _cp1_off = _cl1->offset_center();
        _id = _next_id++, _value = 0, _grad0.fill(RealType(0)), _grad1.fill(RealType(0));
    }

    void evaluate_impl() {
        Point cp0 = _c0->point(), cp1 = _c1->point();
        RealType theta0 = _c0->theta(), theta1 = _c1->theta();
        RealType theta0_f = _c0->theta_f(), theta1_f = _c1->theta_f();

        RealType cos_theta0 = std::cos(theta0), sin_theta0 = std::sin(theta0);
        RealType cos_theta1 = std::cos(theta1), sin_theta1 = std::sin(theta1);
        RealType cos_thetaf0 = std::cos(theta0_f), cos_thetaf1 = std::cos(theta1_f);

        Point _cp0_off_f = _cp0_off * RPoint(cos_thetaf0, 1), _cp1_off_f = _cp1_off * RPoint(cos_thetaf1, 1);
        cp0 = cp0 + _cp0_off_f * RPoint(1, 1) * cos_theta0 + _cp0_off_f.reverse() * RPoint(-1, 1) * sin_theta0;
        cp1 = cp1 + _cp1_off_f * RPoint(1, 1) * cos_theta1 + _cp1_off_f.reverse() * RPoint(-1, 1) * sin_theta1;
        _value = (cp0 - cp1).norm1();
    }

    void gradient_impl() {
        _grad0.fill(RealType(0)), _grad1.fill(RealType(0));
        RealType gamma = _params[0];

        Point cp0 = _c0->point(), cp1 = _c1->point();
        RealType theta0 = _c0->theta(), theta1 = _c1->theta();
        RealType thetaf0 = _c0->theta_f(), thetaf1 = _c1->theta_f();
        RealType cos_theta0 = std::cos(theta0), sin_theta0 = std::sin(theta0);
        RealType cos_theta1 = std::cos(theta1), sin_theta1 = std::sin(theta1);
        RealType cos_thetaf0 = std::cos(thetaf0), sin_thetaf0 = std::sin(thetaf0);
        RealType cos_thetaf1 = std::cos(thetaf1), sin_thetaf1 = std::sin(thetaf1);

        Point _cp0_off_f = _cp0_off * RPoint(cos_thetaf0, 1), _cp1_off_f = _cp1_off * RPoint(cos_thetaf1, 1);
        cp0 = cp0 + _cp0_off_f * RPoint(1, 1) * cos_theta0 + _cp0_off_f.reverse() * RPoint(-1, 1) * sin_theta0;
        cp1 = cp1 + _cp1_off_f * RPoint(1, 1) * cos_theta1 + _cp1_off_f.reverse() * RPoint(-1, 1) * sin_theta1;

        Point grad_0_1 = (cp0 - cp1) * 2;
        _grad0[0] =  _params[1]*grad_0_1.first(), _grad0[1] =  _params[1]*grad_0_1.second();
        _grad1[0] = -_params[1]*grad_0_1.first(), _grad1[1] = -_params[1]*grad_0_1.second();

        RealType pW_px0 =  grad_0_1.first(), pW_py0 =  grad_0_1.second(), pW_pz0 =  2 * _cp0_off.first() * sin_thetaf0; // wl increase
        RealType pW_px1 = -grad_0_1.first(), pW_py1 = -grad_0_1.second(), pW_pz1 =  2 * _cp1_off.first() * sin_thetaf1;

        RealType px0_pthetaf0 = - _cp0_off.first() * cos_theta0 * sin_thetaf0;
        RealType py0_pthetaf0 = - _cp0_off.first() * sin_theta0 * sin_thetaf0;
        RealType pz0_pthetaf0 =   _cp0_off.first() * cos_thetaf0;
        RealType px1_pthetaf1 = - _cp1_off.first() * cos_theta1 * sin_thetaf1;
        RealType py1_pthetaf1 = - _cp1_off.first() * sin_theta1 * sin_thetaf1;
        RealType pz1_pthetaf1 =   _cp1_off.first() * cos_thetaf1;

        RealType pW_pthetaf0 = pW_px0 * px0_pthetaf0 + pW_py0 * py0_pthetaf0 + pW_pz0 * pz0_pthetaf0;
        RealType pW_pthetaf1 = pW_px1 * px1_pthetaf1 + pW_py1 * py1_pthetaf1 + pW_pz1 * pz1_pthetaf1;

        RealType px0_ptheta0 = (RPoint(-sin_theta0*cos_thetaf0, -cos_theta0) * _cp0_off).sum();
        RealType py0_ptheta0 = (RPoint( cos_theta0*cos_thetaf0, -sin_theta0) * _cp0_off).sum();
        RealType px1_ptheta1 = (RPoint(-sin_theta1*cos_thetaf1, -cos_theta1) * _cp1_off).sum();
        RealType py1_ptheta1 = (RPoint( cos_theta1*cos_thetaf1, -sin_theta1) * _cp1_off).sum();

        RealType pW_ptheta0 = pW_px0 * px0_ptheta0 + pW_py0 * py0_ptheta0;
        RealType pW_ptheta1 = pW_px1 * px1_ptheta1 + pW_py1 * py1_ptheta1;

        std::array<RealType, utils::NUM_DISCRETE_VAR>& z0 = _c0->z();
        std::array<RealType, utils::NUM_DISCRETE_VAR>& z1 = _c1->z();
        std::array<RealType, utils::NUM_DISCRETE_VAR> z0_exp, z1_exp;
        if (utils::USE_GUMBEL) {
            z0_exp = utils::gumbel_softmax_with_noise<RealType, utils::NUM_DISCRETE_VAR>(z0, gamma, _c0->gumbel_noise());
            z1_exp = utils::gumbel_softmax_with_noise<RealType, utils::NUM_DISCRETE_VAR>(z1, gamma, _c1->gumbel_noise());
        } else {
            z0_exp = utils::softmax<RealType, utils::NUM_DISCRETE_VAR>(z0, gamma);
            z1_exp = utils::softmax<RealType, utils::NUM_DISCRETE_VAR>(z1, gamma);
        }

        for (IndexType i=0; i<utils::NUM_DISCRETE_VAR; i++) { 
            _grad0[i+2] += _params[2] * pW_ptheta0  * (z0_exp[i] / gamma) * (utils::LEGAL_THETA[i]   - theta0 ); 
            _grad1[i+2] += _params[2] * pW_ptheta1  * (z1_exp[i] / gamma) * (utils::LEGAL_THETA[i]   - theta1 ); 
            _grad0[i+2] += _params[4] * pW_pthetaf0 * (z0_exp[i] / gamma) * (utils::LEGAL_THETA_F[i] - thetaf0); 
            _grad1[i+2] += _params[4] * pW_pthetaf1 * (z1_exp[i] / gamma) * (utils::LEGAL_THETA_F[i] - thetaf1); 
        }
        for (IndexType i=0; i<utils::NUM_DISCRETE_VAR; i++) { 

        }
    }

    void accumulate_gradient_impl() { _c0->acc_grad(_grad0), _c1->acc_grad(_grad1); }

    void cuda_hpwldata_generate() {
        _cuda_hpwl_data.c0_idx = _c0->id();
        _cuda_hpwl_data.c1_idx = _c1->id();
        _cuda_hpwl_data.off_x0 = _cl0->offset_center().first();
        _cuda_hpwl_data.off_y0 = _cl0->offset_center().second();
        _cuda_hpwl_data.off_x1 = _cl1->offset_center().first();
        _cuda_hpwl_data.off_y1 = _cl1->offset_center().second();
        _cuda_hpwl_data.param_row = 0;
    }

    IndexType id() { return _id; }
    RealType value() { return _value; }
    HpwlGrad grad0() { return _grad0; }
    HpwlGrad grad1() { return _grad1; }

    Chiplet* chiplet0() { return _c0; }
    Chiplet* chiplet1() { return _c1; }
    ClusterType* cluster0() { return _cl0; }
    ClusterType* cluster1() { return _cl1; }
    Point offset0() { return _cp0_off; }
    Point offset1() { return _cp1_off; }

    CUDA_HpwlData& cuda_hpwl_data() { return _cuda_hpwl_data; }


private:
    // variable defined in class Caculator: _params: [gamma]
    Chiplet* _c0 = nullptr;
    Chiplet* _c1 = nullptr;
    ClusterType* _cl0 = nullptr;
    ClusterType* _cl1 = nullptr;
    Point _cp0_off;
    Point _cp1_off;
    

    IndexType _id;
    static IndexType _next_id;

    CUDA_HpwlData _cuda_hpwl_data;

    RealType _value; // wirelength consider rotation, evaluate with norm1/norm2
    HpwlGrad _grad0; 
    HpwlGrad _grad1; // x, y, z0, z1, z2, z3
};

class BdyCalculator : public Calculator<BdyCalculator, BdyInput, BdyParams> {
public:
    BdyCalculator(const BdyInput& input, const BdyParams& param) : Calculator(input, param) { 
        auto [c, pbdy, sizebdy] = input;
        _c = c; _bdybox.setLoc(pbdy); _bdybox.setSize(sizebdy);
        _id = _next_id++, _value = 0, _grad.fill(RealType(0));

        _corners[0].setFirst( _c->w()/2), _corners[0].setSecond( _c->h()/2);
        _corners[1].setFirst(-_c->w()/2), _corners[1].setSecond( _c->h()/2);
        _corners[3].setFirst(-_c->w()/2), _corners[3].setSecond(-_c->h()/2);
        _corners[2].setFirst( _c->w()/2), _corners[2].setSecond(-_c->h()/2);
    }

    void evaluate_impl() {
        RealType gamma = _params[0];
        RealType theta = _c->theta();
        RealType cos_2theta = 0.5*(1 + std::cos(2*theta));
        RealType w = cos_2theta*_c->w() + (1-cos_2theta)*_c->h();
        RealType h = cos_2theta*_c->h() + (1-cos_2theta)*_c->w();
        Box cbox = Box(_c->xc(), _c->yc(), w, h);

        _value = static_cast<RealType>(cbox.area() - overlapXY<Point>(&cbox, &_bdybox).absmul());
    }

    void gradient_impl() {
        _grad.fill(RealType(0));
        std::array<LocType, 4> bdy; // xl, yl, xh, yh
        bdy[0] = LOC_TYPE_MAX, bdy[1] = LOC_TYPE_MAX, bdy[2] = LOC_TYPE_MIN, bdy[3] = LOC_TYPE_MIN;
        RealType theta = _c->theta();
        RealType cos_theta = std::cos(theta), sin_theta = std::sin(theta);
        for (auto corner : _corners) {
            Point p = _c->point() + corner * RPoint(1, 1) * cos_theta + corner.reverse() * RPoint(-1, 1) * sin_theta;
            bdy[0] = std::min(bdy[0], p.first()), bdy[1] = std::min(bdy[1], p.second());
            bdy[2] = std::max(bdy[2], p.first()), bdy[3] = std::max(bdy[3], p.second());
        }
        _grad[0] += std::min((bdy[0] - (_bdybox.xc() - _bdybox.w()/2)), static_cast<LocType>(0));
        _grad[1] += std::min((bdy[1] - (_bdybox.yc() - _bdybox.h()/2)), static_cast<LocType>(0));
        _grad[0] += std::max((bdy[2] - (_bdybox.xc() + _bdybox.w()/2)), static_cast<LocType>(0));
        _grad[1] += std::max((bdy[3] - (_bdybox.yc() + _bdybox.h()/2)), static_cast<LocType>(0));

        _grad[0] *= 100*std::abs(_grad[0]);
        _grad[1] *= 100*std::abs(_grad[1]);
    }

    void accumulate_gradient_impl() {
        for (IndexType i = 0; i < _grad.size(); ++i) { 
            _grad[i] *= _params[1]; 
        }
        _c->acc_grad(_grad);
    }

    void cuda_bdydata_generate() {
        _cuda_bdy_data.c_idx = _c->id();
        _cuda_bdy_data.xr = _bdybox.xc(), _cuda_bdy_data.yr = _bdybox.yc();
        _cuda_bdy_data.wr = _bdybox.w() , _cuda_bdy_data.hr = _bdybox.h();
        _cuda_bdy_data.param_row = 1;
    }

    IndexType id() { return _id; }
    RealType value() { return _value; }
    BdyGrad  grad()  { return _grad;  }

    Chiplet* chiplet() { return _c; }
    Box box() { return _bdybox; }

    CUDA_BdyData& cuda_bdy_data() { return _cuda_bdy_data; }
private:
    Chiplet* _c = nullptr;
    Box _bdybox;
    std::array<Point, 4> _corners;

    IndexType _id;
    static IndexType _next_id;

    CUDA_BdyData _cuda_bdy_data;

    RealType _value;
    BdyGrad _grad;
};

class OvlCalculator : public Calculator<OvlCalculator, OvlInput, OvlParams> {
public:
    OvlCalculator(const OvlInput& input, const OvlParams& param) : Calculator(input, param) {
        auto [c0, c1] = input; _c0 = c0; _c1 = c1;
        _id = _next_id++, _value = 0, _grad0.fill(RealType(0)), _grad1.fill(RealType(0));
    }

    void evaluate_impl() {
        RealType gamma = _params[0];
        RealType theta0 = _c0->theta(), theta1 = _c1->theta();
        RealType cos_2theta0 = 0.5*(1+std::cos(2*theta0)), cos_2theta1 = 0.5*(1+std::cos(2*theta1));
        RealType x0 = _c0->xc(), y0 = _c0->yc(), x1 = _c1->xc(), y1 = _c1->yc();
        RealType w0 = cos_2theta0*_c0->w() + (1-cos_2theta0)*_c0->h();
        RealType h0 = cos_2theta0*_c0->h() + (1-cos_2theta0)*_c0->w();
        RealType w1 = cos_2theta1*_c1->w() + (1-cos_2theta1)*_c1->h();
        RealType h1 = cos_2theta1*_c1->h() + (1-cos_2theta1)*_c1->w();

        RealType ovlx = gamma * log(1/(exp(-(x0-x1+w0/2+w1/2)/gamma) + exp(-(x1-x0+w0/2+w1/2)/gamma)) + 1);
        RealType ovly = gamma * log(1/(exp(-(y0-y1+h0/2+h1/2)/gamma) + exp(-(y1-y0+h0/2+h1/2)/gamma)) + 1);

        _value = ovlx * ovly;

    }

    void gradient_impl() {
        _grad0.fill(RealType(0)), _grad1.fill(RealType(0));
        RealType gamma = _params[0];
        RealType theta0 = _c0->theta(), theta1 = _c1->theta();
        RealType cos_2theta0 = 0.5*(1+std::cos(2*theta0)), cos_2theta1 = 0.5*(1+std::cos(2*theta1));
        RealType x0 = _c0->xc(), y0 = _c0->yc(), x1 = _c1->xc(), y1 = _c1->yc();
        RealType w0 = cos_2theta0*_c0->w() + (1-cos_2theta0)*_c0->h();
        RealType h0 = cos_2theta0*_c0->h() + (1-cos_2theta0)*_c0->w();
        RealType w1 = cos_2theta1*_c1->w() + (1-cos_2theta1)*_c1->h();
        RealType h1 = cos_2theta1*_c1->h() + (1-cos_2theta1)*_c1->w();

        RealType exa = exp(-(x0-x1+w0/2+w1/2)/gamma), exb = exp(-(x1-x0+w0/2+w1/2)/gamma);
        RealType eya = exp(-(y0-y1+h0/2+h1/2)/gamma), eyb = exp(-(y1-y0+h0/2+h1/2)/gamma);

        RealType ovlx = gamma * log(1/(exa + exb) + 1);
        RealType ovly = gamma * log(1/(eya + eyb) + 1);

        RealType dx0 = - gamma * ovly * (-exa + exb) / ((1/(exa + exb) + 1) * (exa + exb) * (exa + exb));
        RealType dy0 = - gamma * ovlx * (-eya + eyb) / ((1/(eya + eyb) + 1) * (eya + eyb) * (eya + eyb));
        RealType dx1 = -dx0, dy1 = -dy0;
        _grad0[0] = dx0, _grad0[1] = dy0; 
        _grad1[0] = dx1, _grad1[1] = dy1;
    }

    void accumulate_gradient_impl() {
        for (IndexType i = 0; i < _grad0.size(); ++i) { 
            _grad0[i] *= _params[1], _grad1[i] *= _params[1]; 
        }
        _c0->acc_grad(_grad0), _c1->acc_grad(_grad1);
    }

    void cuda_ovldata_generate() {
        _cuda_ovl_data.c0_idx = _c0->id(), _cuda_ovl_data.c1_idx = _c1->id();
        _cuda_ovl_data.param_row = 2;
    }

    IndexType id() { return _id; }

    CUDA_OvlData& cuda_ovl_data() { return _cuda_ovl_data; }

    RealType value() { return _value; }
    OvlGrad grad0()   { return _grad0;  }
    OvlGrad grad1()   { return _grad1;  }

    Chiplet* chiplet0() { return _c0; }
    Chiplet* chiplet1() { return _c1; }
private:
    Chiplet* _c0 = nullptr;
    Chiplet* _c1 = nullptr;

    IndexType _id;
    static IndexType _next_id;

    CUDA_OvlData _cuda_ovl_data;

    RealType _value;
    OvlGrad _grad0;
    OvlGrad _grad1;
};

class CraCalculator : public Calculator<CraCalculator, CraInput, CraParams> {
public:
    CraCalculator(const CraInput& input, const CraParams& params) : Calculator(input, params) {
        auto [c, r, a] = input; 
        _c = c, _r = r, _Amin = a;
        _id = _next_id++, _value = 0, _grad.fill(RealType(0));
    }

    void evaluate_impl() {
        RealType gamma = _params[0];
        RealType theta = _c->theta();
        RealType cos_2theta = 0.5*(1+std::cos(2*theta));
        RealType xc = _c->xc(), yc = _c->yc();
        RealType wc = cos_2theta*_c->w() + (1-cos_2theta)*_c->h();
        RealType hc = cos_2theta*_c->h() + (1-cos_2theta)*_c->w();
        RealType xr = _r->xc(), yr = _r->yc(), wr = _r->w(), hr = _r->h();

        RealType ovlx = gamma * log(1/(exp(-(xc-xr+wc/2+wr/2)/gamma) + exp(-(xr-xc+wc/2+wr/2)/gamma)) + 1);
        RealType ovly = gamma * log(1/(exp(-(yc-yr+hc/2+hr/2)/gamma) + exp(-(yr-yc+hc/2+hr/2)/gamma)) + 1);
        RealType ovl  = ovlx * ovly;

        RealType eovl_term = (ovl - _Amin) / gamma;
        if (eovl_term >= 10) { _value = 2*_Amin; return; }
        _value = 2*_Amin - ovl + gamma*log(1+exp(eovl_term));
    }

    void gradient_impl() {
        RealType gamma = _params[0];
        RealType theta = _c->theta();
        RealType cos_2theta = 0.5*(1+std::cos(2*theta));
        RealType xc = _c->xc(), yc = _c->yc();
        RealType wc = cos_2theta*_c->w() + (1-cos_2theta)*_c->h();
        RealType hc = cos_2theta*_c->h() + (1-cos_2theta)*_c->w();
        RealType xr = _r->xc(), yr = _r->yc(), wr = _r->w(), hr = _r->h();

        RealType exa = exp(-(xc-xr+wc/2+wr/2)/gamma), exb = exp(-(xr-xc+wc/2+wr/2)/gamma);
        RealType eya = exp(-(yc-yr+hc/2+hr/2)/gamma), eyb = exp(-(yr-yc+hc/2+hr/2)/gamma);

        RealType ovlx = gamma * log(1/(exa + exb) + 1);
        RealType ovly = gamma * log(1/(eya + eyb) + 1);
        RealType ovl  = ovlx * ovly;

        RealType eovl_term = (ovl - _Amin) / gamma;
        if (eovl_term >= 10) { _grad[0] = 0, _grad[1] = 0; return; }
        RealType eovl = exp(eovl_term);

        RealType pA_pxc = - ovly * (-exa + exb) / ((1 + exa + exb) * (exa + exb));
        RealType pA_pyc = - ovlx * (-eya + eyb) / ((1 + eya + eyb) * (eya + eyb));
        RealType norm = std::sqrt(pA_pxc * pA_pxc + pA_pyc * pA_pyc);
        if (norm > static_cast<RealType>(0)) { pA_pxc /= norm; pA_pyc /= norm; }
        RealType k;
        if (_Amin > ovl) { k = std::sqrt(_Amin - ovl); } else { k = 1; }
        RealType dxc = k * (-1 / (eovl + 1)) * pA_pxc;
        RealType dyc = k * (-1 / (eovl + 1)) * pA_pyc;

        _grad[0] = dxc, _grad[1] = dyc;

    }

    void accumulate_gradient_impl() {
        for (IndexType i = 0; i < _grad.size(); ++i) { 
            _grad[i] *= _params[1]; 
        }
        _c->acc_grad(_grad);
    }

    void cuda_cradata_generate() {
        _cuda_cra_data.c_idx = _c->id();
        _cuda_cra_data.xr = _r->xc(), _cuda_cra_data.yr = _r->yc();
        _cuda_cra_data.wr = _r->w() , _cuda_cra_data.hr = _r->h();
        _cuda_cra_data.Amin = _Amin;
        _cuda_cra_data.param_row = 3;
    }

    IndexType id() { return _id; }
    RealType value() { return _value; }
    CraGrad grad() { return _grad; }

    CUDA_CraData& cuda_cra_data() { return _cuda_cra_data; }

    Chiplet* chiplet() { return _c; }
    Reticle* reticle() { return _r; }
    LocType  Amin()    { return _Amin; }
    
private:
    Chiplet* _c;
    Reticle* _r;
    LocType _Amin;

    IndexType _id;
    static IndexType _next_id;

    CUDA_CraData _cuda_cra_data;
    
    RealType _value;
    CraGrad _grad;
};


class GPParamUpdater {
public:
    GPParamUpdater(RealType* param, RealType target, RealType rate) : _param(param), _target(target), _rate(rate) {}
    void update() { 
        if (_param) { 
            *_param = _target + (*_param - _target) * _rate; 
        } 
    }

private:
    RealType* _param;
    RealType _target;
    RealType _rate;
};


class GPlacer {
public:
    GPlacer(std::string workdir, std::string casedir, Database& db, Json::Value& json_gpopt) : _workdir(workdir), _casedir(casedir), _db(db) {
        _pupdaters.reserve(10); // 4+2+2+2
        // Json::Value _json_gpopt = utils::ReadJSON(_workdir+_casedir+"strategy.json");
        _json_strategy = json_gpopt["GPStage"];
        for (IndexType i=0; i<_json_strategy.size(); i++) { 
            if (_json_strategy[i]["id"].asInt() != i) {
                throw std::invalid_argument("some error in json");
            } 
        }

        _param_map["hpwl_smoothness"] = &_hpwl_params[0]; // _hpwl_params[0] -> hpwl_smoothness
        _param_map["hpwl_w_xy"]       = &_hpwl_params[1]; // _hpwl_params[1] -> hpwl_w_xy
        _param_map["hpwl_w_rot"]      = &_hpwl_params[2]; // _hpwl_params[2] -> hpwl_w_rot
        _param_map["hpwl_gumbel"]     = &_hpwl_params[3]; // _hpwl_params[3] -> hpwl_gumbel
        _param_map["hpwl_w_flip"]     = &_hpwl_params[4]; // _hpwl_params[4] -> hpwl_w_flip

        _param_map["bdy_smoothness"]  = &_bdy_params[0];  // _bdy_params[0]  -> bdy_smoothness
        _param_map["bdy_w_xy"]        = &_bdy_params[1];  // _bdy_params[1]  -> bdy_w_xy

        _param_map["ovl_smoothness"]  = &_ovl_params[0];  // _ovl_params[0]  -> ovl_smoothness
        _param_map["ovl_w_xy"]        = &_ovl_params[1];  // _ovl_params[1]  -> ovl_w_xy

        _param_map["cra_smoothness"]  = &_cra_params[0];  // _cra_params[0]  -> cra_smoothness
        _param_map["cra_w_xy"]        = &_cra_params[1];  // _cra_params[1]  -> cra_w_xy
        
        // cuda params map
        _cuda_param_map["hpwl_smoothness"] = std::make_tuple(0, 0); // _hpwl_params[0] -> hpwl_smoothness
        _cuda_param_map["hpwl_w_xy"]       = std::make_tuple(0, 1); // _hpwl_params[1] -> hpwl_w_xy
        _cuda_param_map["hpwl_w_rot"]      = std::make_tuple(0, 2); // _hpwl_params[2] -> hpwl_w_rot
        _cuda_param_map["hpwl_gumbel"]     = std::make_tuple(0, 3); // _hpwl_params[3] -> hpwl_gumbel
        _cuda_param_map["hpwl_w_flip"]     = std::make_tuple(0, 4); // _hpwl_params[4] -> hpwl_w_flip

        _cuda_param_map["bdy_smoothness"]  = std::make_tuple(1, 0);  // _bdy_params[0]  -> bdy_smoothness
        _cuda_param_map["bdy_w_xy"]        = std::make_tuple(1, 1);  // _bdy_params[1]  -> bdy_w_xy

        _cuda_param_map["ovl_smoothness"]  = std::make_tuple(2, 0);  // _ovl_params[0]  -> ovl_smoothness
        _cuda_param_map["ovl_w_xy"]        = std::make_tuple(2, 1);  // _ovl_params[1]  -> ovl_w_xy

        _cuda_param_map["cra_smoothness"]  = std::make_tuple(3, 0);  // _cra_params[0]  -> cra_smoothness
        _cuda_param_map["cra_w_xy"]        = std::make_tuple(3, 1);  // _cra_params[1]  -> cra_w_xy

        _param_logger = spdlog::basic_logger_mt("param_logger", _workdir+_casedir+"placer/log_"+utils::LOG_TIME+"/nlp_iteration/params.txt");
        _param_logger->set_pattern("%v");
        _param_logger->info("iter\thpwl0\tbdy0\tovl0\tcra0\thpwl1\thpwl2\tbdy1\tovl1\tcra1");

    }

    void ParametersInit(bool en_opt, bool en_calc, bool en_var, bool is_var_rand);
    void OptimParametersInit();
    void OperatorInit();
    void VariableInit(bool is_random);
    void CUDAInit();

    void OperatorClear() { _v_hpwl_calc.clear(); _v_bdy_calc.clear(); _v_ovl_calc.clear(); _v_cra_calc.clear(); }
    void addHpwlCalc(ClusterType* cl0, ClusterType* cl1) { _v_hpwl_calc.emplace_back(HpwlCalculator(HpwlInput(cl0, cl1), _hpwl_params)); }
    void addBdyCalc(Chiplet* c, Point p, Size s)         { _v_bdy_calc.emplace_back(BdyCalculator(BdyInput(c, p, s), _bdy_params));}
    void addOvlCalc(Chiplet* c0, Chiplet* c1)            { _v_ovl_calc.emplace_back(OvlCalculator(OvlInput(c0, c1), _ovl_params)); }
    void addCraCalc(Chiplet* c, Reticle* r, LocType a)   { _v_cra_calc.emplace_back(CraCalculator(CraInput(c, r, a), _cra_params)); }

    void GreedyOptimization();
    
    void ValueCalculation();
    void GradientCalculation();
    RealType ValueAccumulation();
    void GradientAccumulation();
    // gpu
    RealType GpuValueGradientCalculation(const arma::vec& x_in, arma::vec* grad_out);
    void CUDA_UpdateHostChiplet();

    void ParametersUpdate();

    double ObjFunc(const arma::vec& x, arma::vec* grad_out, void* opt_data);
    void AdamOptimize(arma::vec& x_out, IndexType stage);
    void BfgsOptimize();

    bool CustomizationChiplet(LocMatrix adj);
    VecLocPair getReticlePathPairs(IndexType id0, IndexType id1, const LocMatrix& adj);

    void dbgtest_value_grad();

    void print_operator_msg(NameType path);
    void print_geometric_msg() { _print_geometric_msg(_workdir+_casedir, _iter++); };
    void _print_geometric_msg(NameType path, IndexType idx);
    void print_val_grad_msg(NameType path, IndexType idx);

private:
    std::string _workdir;
    std::string _casedir;
    Database& _db;
    Json::Value _json_strategy;
    // optimization parameter
    HpwlParams _hpwl_params;
    BdyParams _bdy_params;
    OvlParams _ovl_params;
    CraParams _cra_params;
    // optimization calculator
    std::vector<HpwlCalculator> _v_hpwl_calc;
    std::vector<BdyCalculator> _v_bdy_calc;
    std::vector<OvlCalculator>  _v_ovl_calc;
    std::vector<CraCalculator>  _v_cra_calc;
    // gpu
    CUDA_DataPackage  _cuda_host_data;
    CUDA_DataPackage* _cuda_device_data;
    std::unordered_map<std::string, std::tuple<IndexType, IndexType>> _cuda_param_map;
    // parameter updater
    std::vector<GPParamUpdater> _pupdaters;
    std::unordered_map<std::string, RealType*> _param_map;
    // optimlib parameter
    IndexType _max_iter;
    static IndexType _iter;
    arma::vec _x_init;
    std::shared_ptr<spdlog::logger> _param_logger;
};






} // end namespace

#endif

