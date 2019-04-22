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

namespace ncg {

namespace {
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
        const typename DType<DT>::cctype &a,
        const typename DType<DT>::cctype &b,
        typename DType<DT>::cctype &c
    ) {
        switch (OpType) {
            case BinaryOpKernelType::Add: c = a + b; break;
            case BinaryOpKernelType::Sub: c = a - b; break;
            case BinaryOpKernelType::Mul: c = a * b; break;
            case BinaryOpKernelType::Div: c = a / b; break;
        }
    }
};
} /* !namespace <anonymous> */

class OpAdd : public BinaryElemWiseOp<BinaryOpKernel<BinaryOpKernelType::Add>> {
public:
    NCG_DEF_OPNAME(OpAdd);
};

class OpSub : public BinaryElemWiseOp<BinaryOpKernel<BinaryOpKernelType::Sub>> {
public:
    NCG_DEF_OPNAME(OpSub);
};

class OpMul : public BinaryElemWiseOp<BinaryOpKernel<BinaryOpKernelType::Mul>>{
public:
    NCG_DEF_OPNAME(OpMul);
};

class OpDiv : public BinaryElemWiseOp<BinaryOpKernel<BinaryOpKernelType::Div>>{
public:
    NCG_DEF_OPNAME(OpDiv);
};

} /* !namespace ncg */

#endif /* !ARITH_H */
