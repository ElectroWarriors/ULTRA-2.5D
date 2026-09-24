#pragma once

#include<iostream>
#include <fstream>
#include <sstream>
#include<string>
#include<vector>
#include<cstdint>  // UINT32_MAX etc.

#include<armadillo>

using IndexType  = std::uint32_t;
using IntType    = std::int32_t;
using RealType   = double;
using Byte       = std::uint8_t;
using LocType    = double; // std::int64_t;
using BoolType   = Byte;
using NameType   = std::string;
// using LocXYType  = std::pair<LocType, LocType>;
using LocMatrix  = arma::imat; // std::vector<std::vector<LocType>>;
using VecLocPair = std::vector<std::pair<IndexType, IndexType>>;

// routability.h
using CoordId = std::pair<IntType, IntType>;
using Path  = std::vector<CoordId>;

constexpr IndexType INDEX_TYPE_MAX  = UINT32_MAX;
constexpr IntType INT_TYPE_MAX      = INT32_MAX;
constexpr IntType INT_TYPE_MIN      = INT32_MIN;
constexpr RealType REAL_TYPE_MAX    = 1e100;
constexpr RealType REAL_TYPE_MIN    = -1e100;
constexpr RealType REAL_TYPE_TOL    = 1e-6;
constexpr LocType LOC_TYPE_MAX      = INT64_MAX;
constexpr LocType LOC_TYPE_MIN      = INT64_MIN;
constexpr RealType PI = 3.14159; 