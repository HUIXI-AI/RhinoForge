#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cstdint>

struct ValuOpInfo {
    int op;
    int custom_op;
    const char* info;
    bool unary;
};

// -----------------------------------------------------------------------------
// 枚举定义 - 值必须连续从 0 开始，用于数组索引
// -----------------------------------------------------------------------------
enum class ValuOpType : uint8_t {
    ADD = 0, SUB, MUL, DIV, MAX, MIN, ACC, RECIPROCAL, SQRT, SQRT_RECIPROCAL,
    SIN, COS, MAX_POOL, POWER2, LOG, EXP, EXP_ACC, EXP_DIV, MUL_SEL,
    SIGMOID, TANH, ACC_DIV, MUL2_ADD1_ACC_IMM012, MUL2_ADD1_ACC_IMM013,
    LT, LTE, GT, GTE, EQ, NEQ, POW,
    FLOOR, CEIL, CLIP, SIGMOID_Q, TANH_Q, TANH_FIX,
    SILU, HARDSIGMOID, SOFTPLUS, ABS,
    _COUNT  // 用于获取枚举数量
};

// -----------------------------------------------------------------------------
// 主表: 运算符定义 - 使用数组替代 unordered_map，O(1) 查找
// -----------------------------------------------------------------------------
static constexpr ValuOpInfo valu_op_array[] = {
    {0, 0, "scale_r * (scale_a * a + scale_b * b)", false},           // ADD
    {1, 0, "scale_r * (scale_a * a - scale_b * b)", false},           // SUB
    {2, 0, "scale_r * (scale_a * a * scale_b * b)", false},           // MUL
    {3, 0, "scale_r * (scale_a * a / scale_b * b)", false},           // DIV
    {4, 0, "scale_r * max(scale_a * a, scale_b * b)", false},         // MAX
    {5, 0, "scale_r * min(scale_a * a, scale_b * b)", false},         // MIN
    {6, 0, "scale_r * acc(scale_a * a)", true},                       // ACC
    {7, 0, "scale_r * (1 / scale_a * a)", true},                      // RECIPROCAL
    {8, 0, "scale_r * sqrt(scale_a * a)", true},                      // SQRT
    {9, 0, "scale_r * (1 / sqrt(scale_a * a))", true},                // SQRT_RECIPROCAL
    {10, 0, "scale_r * sin(scale_a * a)", true},                      // SIN
    {11, 0, "scale_r * cos(scale_a * a)", true},                      // COS
    {12, 0, "scale_r * max_pool(scale_a * a)", true},                 // MAX_POOL
    {13, 0, "scale_r * power2(scale_a * a)", true},                   // POWER2
    {14, 0, "scale_r * log(scale_a * a)", true},                      // LOG
    {15, 0, "scale_r * exp(scale_a * a)", true},                      // EXP
    {16, 0, "scale_r * exp_acc(scale_a * a)", true},                  // EXP_ACC
    {17, 0, "scale_r * exp_div(scale_a * a) / b", false},             // EXP_DIV
    {18, 0, "if (a>0) a else imm0 * a", true},                        // MUL_SEL
    {19, 0, "1 /(exp(-a) +1)", true},                                 // SIGMOID
    {20, 0, "tanh", true},                                            // TANH
    {21, 0, "scale_r * acc_div(scale_a * a)", true},                  // ACC_DIV
    {22, 0, "acc(imm2 * (imm0 * a + imm1 * b))", false},              // MUL2_ADD1_ACC_IMM012
    {23, 0, "acc(imm3 * (imm0 * a + imm1 * b))", false},              // MUL2_ADD1_ACC_IMM013
    {31, 32, "less than", false},                                     // LT
    {31, 33, "less than or equal", false},                            // LTE
    {31, 34, "greater than", false},                                  // GT
    {31, 35, "greater than or equal", false},                         // GTE
    {31, 36, "equal", false},                                         // EQ
    {31, 37, "not equal", false},                                     // NEQ
    {31, 38, "pow", false},                                           // POW
    {31, 128, "floor, round to negative infinity", true},             // FLOOR
    {31, 129, "ceil, round to positive infinity", true},              // CEIL
    {31, 130, "clip", true},                                          // CLIP
    {31, 131, "sigmoid support scale_a/scale_r", true},               // SIGMOID_Q
    {31, 132, "tanh support scale_a/scale_r", true},                  // TANH_Q
    {31, 133, "tanh fix", true},                                      // TANH_FIX
    {31, 134, "silu", true},                                          // SILU
    {31, 135, "hardsigmoid", true},                                   // HARDSIGMOID
    {31, 136, "softplus", true},                                      // SOFTPLUS
    {31, 137, "abs", true},                                           // ABS
};

// 编译期检查数组大小与枚举数量一致
static_assert(sizeof(valu_op_array) / sizeof(valu_op_array[0]) == static_cast<size_t>(ValuOpType::_COUNT),
              "valu_op_array size must match ValuOpType::_COUNT");

// -----------------------------------------------------------------------------
// 数据类型配置常量 - 替代 unordered_map 查找
// -----------------------------------------------------------------------------
namespace ValuDtypeCfg {
    constexpr int UINT8 = 0;
    constexpr int INT8 = 1;
    constexpr int FP16 = 2;
    constexpr int BF16 = 3;
    constexpr int FP8_E2M5 = 4;
    constexpr int FP8_E4M3 = 5;
}

// FP16 数据类型配置 - 最常用，预计算好
constexpr uint16_t FP16_RAB_DATA_TYPE_CFG = (ValuDtypeCfg::FP16 << 6) |
                                            (ValuDtypeCfg::FP16 << 3) |
                                            ValuDtypeCfg::FP16;

// -----------------------------------------------------------------------------
// 热点函数 - O(1) 数组索引
// -----------------------------------------------------------------------------
inline const ValuOpInfo& get_valu_op_info(ValuOpType type) {
    return valu_op_array[static_cast<uint8_t>(type)];
}

// -----------------------------------------------------------------------------
// 非热点函数 - 保留 unordered_map 实现
// -----------------------------------------------------------------------------
static const std::unordered_map<std::string, ValuOpType> str_to_enum = {
    {"add", ValuOpType::ADD}, {"sub", ValuOpType::SUB}, {"mul", ValuOpType::MUL},
    {"div", ValuOpType::DIV}, {"max", ValuOpType::MAX}, {"min", ValuOpType::MIN},
    {"acc", ValuOpType::ACC}, {"reciprocal", ValuOpType::RECIPROCAL},
    {"sqrt", ValuOpType::SQRT}, {"sqrt_reciprocal", ValuOpType::SQRT_RECIPROCAL},
    {"sin", ValuOpType::SIN}, {"cos", ValuOpType::COS}, {"max_pool", ValuOpType::MAX_POOL},
    {"power2", ValuOpType::POWER2}, {"log", ValuOpType::LOG}, {"exp", ValuOpType::EXP},
    {"exp_acc", ValuOpType::EXP_ACC}, {"exp_div", ValuOpType::EXP_DIV},
    {"mul_sel", ValuOpType::MUL_SEL}, {"sigmoid", ValuOpType::SIGMOID},
    {"tanh", ValuOpType::TANH}, {"acc_div", ValuOpType::ACC_DIV},
    {"mul2_add1_acc_imm012", ValuOpType::MUL2_ADD1_ACC_IMM012},
    {"mul2_add1_acc_imm013", ValuOpType::MUL2_ADD1_ACC_IMM013},
    {"lt", ValuOpType::LT}, {"lte", ValuOpType::LTE}, {"gt", ValuOpType::GT},
    {"gte", ValuOpType::GTE}, {"eq", ValuOpType::EQ}, {"neq", ValuOpType::NEQ},
    {"pow", ValuOpType::POW},
    {"floor", ValuOpType::FLOOR}, {"ceil", ValuOpType::CEIL}, {"clip", ValuOpType::CLIP},
    {"sigmoid_q", ValuOpType::SIGMOID_Q}, {"tanh_q", ValuOpType::TANH_Q},
    {"tanh_fix", ValuOpType::TANH_FIX}, {"silu", ValuOpType::SILU},
    {"hardsigmoid", ValuOpType::HARDSIGMOID}, {"softplus", ValuOpType::SOFTPLUS},
    {"abs", ValuOpType::ABS},
};

inline ValuOpType valu_op_from_string(const std::string& name) {
    return str_to_enum.at(name);
}

inline const char* valu_op_to_string(ValuOpType type) {
    static const char* names[] = {
        "add", "sub", "mul", "div", "max", "min", "acc", "reciprocal",
        "sqrt", "sqrt_reciprocal", "sin", "cos", "max_pool", "power2",
        "log", "exp", "exp_acc", "exp_div", "mul_sel", "sigmoid", "tanh",
        "acc_div", "mul2_add1_acc_imm012", "mul2_add1_acc_imm013",
        "lt", "lte", "gt", "gte", "eq", "neq", "pow",
        "floor", "ceil", "clip", "sigmoid_q", "tanh_q", "tanh_fix",
        "silu", "hardsigmoid", "softplus", "abs"
    };
    return names[static_cast<uint8_t>(type)];
}

inline std::vector<ValuOpType> get_unary_ops() {
    std::vector<ValuOpType> out;
    for (uint8_t i = 0; i < static_cast<uint8_t>(ValuOpType::_COUNT); i++) {
        if (valu_op_array[i].unary)
            out.push_back(static_cast<ValuOpType>(i));
    }
    return out;
}

inline std::vector<ValuOpType> get_binary_ops() {
    std::vector<ValuOpType> out;
    for (uint8_t i = 0; i < static_cast<uint8_t>(ValuOpType::_COUNT); i++) {
        if (!valu_op_array[i].unary)
            out.push_back(static_cast<ValuOpType>(i));
    }
    return out;
}

// -----------------------------------------------------------------------------
// 数据宽度常量
// -----------------------------------------------------------------------------
namespace DWidth {
    constexpr int UINT8 = 1;
    constexpr int INT8 = 1;
    constexpr int FP16 = 2;
    constexpr int BF16 = 2;
    constexpr int FP8_E2M5 = 1;
    constexpr int FP8_E4M3 = 1;
}
