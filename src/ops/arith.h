/*
 * arith.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#ifndef ARITH_H
#define ARITH_H

#include "core/op.h"

namespace ncg {

class ElemWiseOp : public Op {
public:
    virtual void check_inputs(OpContext &ctx, const TensorVec &inputs) {
        if (inputs.size() == 0) {
            return ;
        }
        for (ssize_t i = 0; i < inputs.size(); ++i) {
            if (!inputs[0]->desc().is_compatible(inputs[i]->desc())) {
                ctx.error(this) << "incompatible inputs. inputs[0] = " << inputs[0]->desc() << "; inputs[" << i << "]=" << inputs[i]->desc() << ".";
            }
        }
    }
};

template <typename OpKernel>
class UnaryElemWiseOp : public ElemWiseOp {
public:
    virtual void check_inputs(OpContext &ctx, const TensorVec &inputs) {
        ElemWiseOp::check_inputs(ctx, inputs);
        if (ctx.is_error()) return;
        if (inputs.size() != 1) {
            ctx.error(this) << "requires 1 input tensor.";
        }
    }

    virtual TensorVec compute(OpContext &, const TensorVec &inputs) {
        TensorPtr output = empty(inputs[0]->desc().dtype(), inputs[0]->desc().shape_vec());

#define UNARY_COMPUTE_DTYPE(dtype) compute_inner_<DTypeName::dtype>(inputs, output)
NCG_SWITCH_DTYPE_ALL(inputs[0]->desc().dtype(), UNARY_COMPUTE_DTYPE);
#undef UNARY_COMPUTE_DTYPE

        return {output};
    }
private:
    template <DTypeName DT>
    void compute_inner_(const TensorVec &inputs, TensorPtr &output) {
        size_t n = inputs[0]->desc().numel();
        auto kernel = OpKernel();
        auto a = inputs[0]->as<DT>();
        auto b = output->as<DT>();
        for (ssize_t i = 0; i < n; ++i) {
            kernel.template compute<DT>(a->elat(i), b->elat(i));
        }
    }
};

template <typename OpKernel>
class BinaryElemWiseOp : public ElemWiseOp {
public:
    virtual void check_inputs(OpContext &ctx, const TensorVec &inputs) {
        ElemWiseOp::check_inputs(ctx, inputs);
        if (ctx.is_error()) return;
        if (inputs.size() != 2) {
            ctx.error(this) << "requires 2 input tensors.";
        }
    }

    virtual TensorVec compute(OpContext &, const TensorVec &inputs) {
        size_t n = inputs[0]->desc().numel();
        TensorPtr output = empty(inputs[0]->desc().dtype(), inputs[0]->desc().shape_vec());

#define BINARY_COMPUTE_DTYPE(dtype) compute_inner_<DTypeName::dtype>(inputs, output)
NCG_SWITCH_DTYPE_ALL(inputs[0]->desc().dtype(), BINARY_COMPUTE_DTYPE);
#undef BINARY_COMPUTE_DTYPE

        return {output};
    }
private:
    template <DTypeName DT>
    void compute_inner_(const TensorVec &inputs, TensorPtr &output) {
        size_t n = inputs[0]->desc().numel();
        auto kernel = OpKernel();
        auto a = inputs[0]->as<DT>();
        auto b = inputs[1]->as<DT>();
        auto c = output->as<DT>();
        for (ssize_t i = 0; i < n; ++i) {
            kernel.template compute<DT>(a->elat(i), b->elat(i), c->elat(i));
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
