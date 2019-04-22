/*
 * arith.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#ifndef ARITH_H
#define ARITH_H

#include "core/op.h"
#include "ops/op_common.h"
#include <cmath>

namespace ncg {

namespace {
enum class UnaryOpKernelType : int {
    Neg,
    Sin,
    Cos,
    Tan,
    Log,
    Exp,
    Tanh,
    Sigmoid,
};

template <UnaryOpKernelType OpType>
struct UnaryOpKernel {
    template <DTypeName DT>
    void compute(
        OpContext &ctx,
        Op *op,
        const typename DType<DT>::cctype &a,
        typename DType<DT>::cctype &b
    ) {
        if (OpType == UnaryOpKernelType::Neg) {
            b = -a;
            return;
        }

        if (DT != DTypeName::Float32 && DT != DTypeName::Float64) {
            ctx.error(op) << "Operator for integer not implemented";
            return;
        }

        switch (OpType) {
            case UnaryOpKernelType::Sin: b = std::sin(a); break;
            case UnaryOpKernelType::Sin: b = std::cos(a); break;
            case UnaryOpKernelType::Sin: b = std::tan(a); break;
            case UnaryOpKernelType::Log:
                if (a <= 0) {
                    ctx.error(op) << "LOG operator's input must be positive";
                } else {
                    b = std::log(a);
                }
                break;
            case UnaryOpKernelType::Exp: b = std::exp(a); break;
            case UnaryOpKernelType::Tanh: b = std::tanh(a); break;
            case UnaryOpKernelType::Sigmoid: b = 1 / (1 + std::exp(-a)); break;
        }
    }
};

enum class BinaryOpKernelType : int {
    Add,
    Sub,
    Mul,
    Div,
};

template <BinaryOpKernelType OpType>
struct BinaryOpKernel {
    template <DTypeName DT>
    void compute(
        OpContext &ctx,
        Op *op,
        const typename DType<DT>::cctype &a,
        const typename DType<DT>::cctype &b,
        typename DType<DT>::cctype &c
    ) {
        switch (OpType) {
            case BinaryOpKernelType::Add: c = a + b; break;
            case BinaryOpKernelType::Sub: c = a - b; break;
            case BinaryOpKernelType::Mul: c = a * b; break;
            case BinaryOpKernelType::Div:
                if (DT != DTypeName::Float32 && DT != DTypeName::Float64) {
                    ctx.error(op) << "Division for integer not implemented";
                    break;
                }
                if (b == 0) {
                    ctx.error(op) << "Division by zero";
                    break;
                }
                c = a / b;
                break;
        }
    }
};
} /* !namespace <anonymous> */

#define DEF_UNARY_ELEMWISE_OP(name) \
class Op##name : public UnaryElemWiseOp<UnaryOpKernel<UnaryOpKernelType::name>> { \
public: \
    NCG_DEF_OPNAME(Op##name); \
}

DEF_UNARY_ELEMWISE_OP(Neg);
DEF_UNARY_ELEMWISE_OP(Sin);
DEF_UNARY_ELEMWISE_OP(Cos);
DEF_UNARY_ELEMWISE_OP(Tan);
DEF_UNARY_ELEMWISE_OP(Log);
DEF_UNARY_ELEMWISE_OP(Exp);
DEF_UNARY_ELEMWISE_OP(Tanh);
DEF_UNARY_ELEMWISE_OP(Sigmoid);

#define DEF_BINARY_ELEMWISE_OP(name) \
class Op##name : public BinaryElemWiseOp<BinaryOpKernel<BinaryOpKernelType::name>> { \
public: \
    NCG_DEF_OPNAME(Op##name); \
}

DEF_BINARY_ELEMWISE_OP(Add);
DEF_BINARY_ELEMWISE_OP(Sub);
DEF_BINARY_ELEMWISE_OP(Mul);
DEF_BINARY_ELEMWISE_OP(Div);

} /* !namespace ncg */

#endif /* !ARITH_H */
