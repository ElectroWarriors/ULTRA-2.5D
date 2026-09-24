#ifndef UTILD_H_
#define UTILD_H_
#pragma once

#include"type.h"

#include <cstddef>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>
#include <cmath>
#include <random>
#include <array>

#include<json/json.h>

namespace utils {

extern RealType XSCALE;
extern RealType YSCALE;
extern RealType WHITESPACE;
extern std::array<RealType, 4> TOTAL_BDY;
extern NameType LOG_TIME;

extern bool EN_GPU;
extern bool EN_DEBUG;
extern bool EN_OUTPUT_DB;
extern bool EN_OUTPUT_OP;
extern bool EN_OUTPUT_GEO;
extern bool EN_OUTPUT_GD;
extern bool EN_OUTPUT_HP;

extern RealType HPWL_PARAM0;
extern RealType BDY_PARAM0;
extern RealType OVL_PARAM0;
extern RealType CRA_PARAM0;

extern RealType MAX_WEIGHT;
extern RealType HPWL_PARAM1_XY;
extern RealType HPWL_PARAM1_ALPHA;
extern RealType HPWL_PARAM1_GUMBEL;
extern RealType BDY_PARAM1;
extern RealType OVL_PARAM1;
extern RealType CRA_PARAM1;

extern bool USE_GUMBEL; 
extern bool USE_FLIP;

extern const RealType epsilon;

inline constexpr std::size_t NUM_DISCRETE_VAR = 8;
extern const std::array<RealType, NUM_DISCRETE_VAR> LEGAL_THETA;
extern const std::array<RealType, NUM_DISCRETE_VAR> LEGAL_THETA_F;

static_assert(std::is_same_v<RealType, double>, "RealType is not double in utils.h");
static_assert(NUM_DISCRETE_VAR == 8, "NUM_DISCRETE_VAR not 8 in utils.h");

// Commonly used global functions
void print();

Json::Value ReadJSON(std::string filepath);

bool WriteJSON(const std::string& filepath, const Json::Value& root);

LocMatrix readIntTXT(const std::string& filename);

void readConnFromJSON(LocMatrix &mat, const Json::Value& root, bool isSparse);



template<typename T, IndexType N>
std::array<T, N> softmax(const std::array<T, N>& z, T gamma) {
    std::array<T, N> w;
    T max_z = *std::max_element(z.begin(), z.end());
    T sum = 0;
    for (IndexType i = 0; i < N; ++i) { w[i] = std::exp((z[i] - max_z) / gamma); sum += w[i]; }
    for (IndexType i = 0; i < N; ++i) { w[i] /= sum; }
    return w;
}

template<typename T, IndexType N>
std::array<T, N> sample_gumbel(RealType k) {
    std::array<T, N> noise;
    static std::mt19937 gen(std::random_device{}());
    static std::uniform_real_distribution<T> dist(T(1e-6), T(1.0 - 1e-6)); // avoid log(0)
    for (IndexType i = 0; i < N; ++i) { T u = dist(gen); noise[i] = -k * std::log(-std::log(u)); }
    return noise;
}

template<typename T, IndexType N>
std::array<T, N> gumbel_softmax_with_noise(const std::array<T, N>& z, T gamma, const std::array<T, N>& noise) {
    std::array<T, N> w;
    T max_z = T(0), sum = T(0);

    for (IndexType i = 0; i < N; ++i) {
        T perturbed = (z[i] + noise[i]) / gamma;
        if (i == 0 || perturbed > max_z) { max_z = perturbed; }
        w[i] = perturbed;
    }

    for (IndexType i = 0; i < N; ++i) { w[i] = std::exp(w[i] - max_z); sum += w[i]; }
    for (IndexType i = 0; i < N; ++i) { w[i] /= sum; }
    return w;
}


class TimerCounter {
public:
    explicit TimerCounter(IndexType count) : _count(count) {}
    bool tick() {
        if (_count <= 0) return true;
        --_count;
        return _count == 0;
    }

    IndexType remaining() const { return _count; }

private:
    IndexType _count;
};



} // end utils


#endif

