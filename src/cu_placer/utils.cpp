
#include<iostream>
#include <fstream>
#include<string>

#include"type.h"
#include"utils.h"
#include"db.h"

#include<json/json.h>
#include"spdlog/spdlog.h"




namespace utils {
// Commonly used global functions

void print() { std::cout << "hello, namespace" << std::endl; }

Json::Value ReadJSON(std::string filepath) {
    spdlog::info("[parser] read json file: {}", filepath);
    Json::Reader reader;
    Json::Value jsondata;
    std::ifstream infile(filepath, std::ios::binary);
    if (!infile.is_open()) {
        spdlog::error("error opening file for reading");
        return 0;
    }
    if (!reader.parse(infile, jsondata)) {
        return 0;
    }
    return jsondata;
}

bool WriteJSON(const std::string& filepath, const Json::Value& root) {
    spdlog::info("write json to path: {}", filepath);
    std::ofstream file(filepath);
    if (file.is_open()) {
        file << root.toStyledString(); // 美化输出
        file.close();
        spdlog::info("json data has been written.");
        return true;
    } else {
        spdlog::error("error opening file for writing");
        return false;
    }
}

LocMatrix readIntTXT(const std::string& filepath) {
    spdlog::info("[parser] loading arma::imat from: {}", filepath);
    arma::imat data;
    if (!data.load(filepath, arma::raw_ascii)) {
        spdlog::error("failed to load matrix from file");
        return LocMatrix();
    }
    return data;
}


// LocMatrix readIntJSON(const Json::Value& root, bool isSparse) {
//     spdlog::info("loading sparse-style arma::imat from Json::Value");

//     if (!root.isArray() || root.empty()) {
//         spdlog::error("input JSON is not a non-empty array");
//         return LocMatrix(); // 空矩阵
//     }

//     if (isSparse) {
//         const IndexType n = root.size();
//         LocMatrix result(n, 3);

//         for (IndexType i = 0; i < n; ++i) {
//             const auto& row = root[i];
//             if (!row.isArray() || row.size() != 3) {
//                 spdlog::error("invalid sparse triplet at index {}", i);
//                 return LocMatrix();
//             }
//             result(i, 0) = row[0].asInt(); // row
//             result(i, 1) = row[1].asInt(); // col
//             result(i, 2) = row[2].asInt(); // value
//         }
//         return result;

//     } else {
//         const IndexType rows = root.size();
//         const IndexType cols = root[0].size();

//         IndexType count = 0;
//         for (IndexType r = 0; r < rows; ++r) {
//             const auto& row = root[r];
//             if (!row.isArray() || row.size() != cols) {
//                 spdlog::error("inconsistent row size at index {}", r);
//                 return LocMatrix();
//             }

//             for (IndexType c = 0; c < cols; ++c) {
//                 if (row[c].asInt() != 0) { ++count; }
//             }
//         }

//         LocMatrix result(count, 3);
//         IndexType idx = 0;

//         for (IndexType r = 0; r < rows; ++r) {
//             const auto& row = root[r];
//             for (IndexType c = 0; c < cols; ++c) {
//                 int val = row[c].asInt();
//                 if (val != 0) {
//                     result(idx, 0) = r;
//                     result(idx, 1) = c;
//                     result(idx, 2) = val;
//                     ++idx;
//                 }
//             }
//         }
//         return result;
//     }
// }

void readConnFromJSON(LocMatrix &mat, const Json::Value& root, bool isSparse) {
    spdlog::info("loading sparse-style arma::imat from Json::Value");

    if (!root.isArray() || root.empty()) { spdlog::error("input JSON is not a non-empty array"); }

    if (isSparse) {
        for (auto row : root) {
            if (!row.isArray() || row.size() != 3) { spdlog::error("invalid sparse row"); }
            mat(row[0].asInt(), row[1].asInt()) = row[2].asInt();
            mat(row[1].asInt(), row[0].asInt()) = row[2].asInt();
        }
    } else {
        if (root.size() != mat.n_rows) { spdlog::info("input rows fit the the number of chiplets. "); }

        for (IndexType r = 0; r < mat.n_rows; r++) {
            const auto& row = root[r];
            if ( row.size() != mat.n_cols) { spdlog::info("input cols fit the the number of chiplets. "); }
            for (IndexType c = 0; c < mat.n_cols; c++) { 
                mat(r, c) = row[c].asInt();
            }
        }
    }
}



std::string getCurrentTimestamp() {
    using namespace std::chrono;

    auto now = system_clock::now();
    std::time_t t = system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d-%H%M%S");
    return oss.str();
}

RealType XSCALE;
RealType YSCALE;
RealType WHITESPACE;
std::array<RealType, 4> TOTAL_BDY;
NameType LOG_TIME = getCurrentTimestamp();

bool EN_GPU = false;
bool EN_DEBUG = true;
bool EN_OUTPUT_DB = false;
bool EN_OUTPUT_OP = false;
bool EN_OUTPUT_GEO = false;
bool EN_OUTPUT_GD = false;
bool EN_OUTPUT_HP = false;
// case 8, 22, 225
RealType HPWL_PARAM0 = 40;
RealType BDY_PARAM0  = 10;
RealType OVL_PARAM0  = 10;
RealType CRA_PARAM0  = 10;
// case 1225
// RealType HPWL_PARAM0 = 4;
// RealType BDY_PARAM0  = 1;
// RealType OVL_PARAM0  = 1;
// RealType CRA_PARAM0  = 1;

RealType MAX_WEIGHT = 4.00;
RealType HPWL_PARAM1_XY = 3.00;
RealType HPWL_PARAM1_ALPHA = 3.00;
RealType HPWL_PARAM1_GUMBEL = 1.00;
RealType BDY_PARAM1 = 4.00;
RealType OVL_PARAM1 = 0.5;
RealType CRA_PARAM1 = 0.00;

bool USE_GUMBEL = true;
bool USE_FLIP = false;

const RealType epsilon = 0.1;

static_assert(NUM_DISCRETE_VAR == 8, "NUM_DISCRETE_VAR not visible");
static_assert(std::is_same_v<RealType, double>, "RealType is not double");

// constexpr IndexType NUM_DISCRETE_VAR = 8;
const std::array<RealType, NUM_DISCRETE_VAR> LEGAL_THETA = {
    RealType(0.0), RealType(PI/2.0), RealType(PI), RealType(3.0*PI/2.0), 
    RealType(0.0), RealType(PI/2.0), RealType(PI), RealType(3.0*PI/2.0)
};
const std::array<RealType, NUM_DISCRETE_VAR> LEGAL_THETA_F = {
    RealType(0.0), RealType(0.0), RealType(0.0), RealType(0.0), 
    RealType(PI) , RealType(PI) , RealType(PI) , RealType(PI)
};


}

