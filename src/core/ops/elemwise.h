/*
 * elemwise.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#ifndef CORE_OPS_ARITH_H
#define CORE_OPS_ARITH_H

#include "core/op.h"

#include <cmath>

namespace ncg {

class OpElemwiseBase : public Op {
public:
    virtual void check_inputs(OpContext &ctx, const TensorVec &inputs) {
        NCG_OP_CHECK_NONEMPTY_INPUTS(ctx, inputs);
        NCG_OP_CHECK_COMPATIBLE_DTYPE(ctx, inputs);
        NCG_OP_CHECK_COMPATIBLE_SHAPE(ctx, inputs);
    }
};

class OpCastDesc : public OpDesc {
public:
    OpCastDesc() : dtype(DTypeName::Int8) {}
    OpCastDesc(DTypeName dtype) : dtype(dtype) {}
    virtual ~OpCastDesc() = default;

    DTypeName dtype;
};

class OpCast : public OpElemwiseBase {
public:
    NCG_OP_DEF_NAME(OpCast);

    virtual void check_inputs(OpContext &ctx, const TensorVec &inputs) {
        OpElemwiseBase::check_inputs(ctx, inputs);
        NCG_OP_CHECK_CTX_CLEAN(ctx);
        NCG_OP_CHECK_NR_INPUTS(ctx, inputs, 1);
    }

    virtual TensorVec compute(OpContext &ctx, const TensorVec &inputs) {
        const auto &desc = this->template desc<OpCastDesc>();
        TensorPtr output = empty(desc.dtype, inputs[0]->desc().shape_vec());

#define CAST_COMPUTE_DTYPE(dtype) kernel_<DTypeName::dtype>(ctx, inputs, output)
NCG_DTYPE_SWITCH_ALL(inputs[0]->desc().dtype(), CAST_COMPUTE_DTYPE);
#undef CAST_COMPUTE_DTYPE

        return {output};
    }

private:
    template <DTypeName DT>
    void kernel_(OpContext &ctx, const TensorVec &inputs, TensorPtr &output) {
        size_t n = inputs[0]->desc().numel();
        auto a = inputs[0]->as<DT>();
        auto b = output->as<DT>();
        for (ssize_t i = 0; i < n; ++i) {
            b->mutable_elat(i) = a->elat(i);
        }
    }
};

class OpCond : public OpElemwiseBase {
public:
    NCG_OP_DEF_NAME(OpCond);

    virtual void check_inputs(OpContext &ctx, const TensorVec &inputs) {
        OpElemwiseBase::check_inputs(ctx, inputs);
        NCG_OP_CHECK_CTX_CLEAN(ctx);
        NCG_OP_CHECK_NR_INPUTS(ctx, inputs, 3);
    }

    virtual TensorVec compute(OpContext &ctx, const TensorVec &inputs) {
        TensorPtr output = empty(inputs[2]->desc().dtype(), inputs[2]->desc().shape_vec());

#define COND_COMPUTE_DTYPE(dtype) compute_inner_<DTypeName::dtype>(ctx, inputs, output)
NCG_DTYPE_SWITCH_ALL(inputs[0]->desc().dtype(), COND_COMPUTE_DTYPE);
#undef COND_COMPUTE_DTYPE

        return {output};
    }

private:
    template <DTypeName DT>
    void compute_inner_(OpContext &ctx, const TensorVec &inputs, TensorPtr &output) {
        size_t n = inputs[0]->desc().numel();
        auto a = inputs[0]->as<DT>();
        auto b = inputs[1]->as<DT>();
        auto c = inputs[2]->as<DT>();
        auto d = output->as<DT>();
        for (ssize_t i = 0; i < n; ++i) {
            d->mutable_elat(i) = a->elat(i) > 0 ? b->elat(i) : c->mutable_elat(i);
        }
    }
};

template <typename OpKernel>
class OpUnaryElemwiseBase : public OpElemwiseBase {
public:
    virtual void check_inputs(OpContext &ctx, const TensorVec &inputs) {
        OpElemwiseBase::check_inputs(ctx, inputs);
        NCG_OP_CHECK_CTX_CLEAN(ctx);
        NCG_OP_CHECK_NR_INPUTS(ctx, inputs, 1);
    }

    virtual TensorVec compute(OpContext &ctx, const TensorVec &inputs) {
        TensorPtr output = empty(inputs[0]->desc().dtype(), inputs[0]->desc().shape_vec());

#define UNARY_COMPUTE_DTYPE(dtype) kernel_<DTypeName::dtype>(ctx, inputs, output)
NCG_DTYPE_SWITCH_ALL(inputs[0]->desc().dtype(), UNARY_COMPUTE_DTYPE);
#undef UNARY_COMPUTE_DTYPE

        return {output};
    }

private:
    template <DTypeName DT>
    void kernel_(OpContext &ctx, const TensorVec &inputs, TensorPtr &output) {
        size_t n = inputs[0]->desc().numel();
        auto kernel = OpKernel();
        auto a = inputs[0]->as<DT>();
        auto b = output->as<DT>();
        for (ssize_t i = 0; i < n; ++i) {
            kernel.template compute<DT>(ctx, this, a->elat(i), b->mutable_elat(i));
        }
    }
};

template <typename OpKernel>
class OpBinaryElemwiseBase : public OpElemwiseBase {
public:
    virtual void check_inputs(OpContext &ctx, const TensorVec &inputs) {
        OpElemwiseBase::check_inputs(ctx, inputs);
        NCG_OP_CHECK_CTX_CLEAN(ctx);
        NCG_OP_CHECK_NR_INPUTS(ctx, inputs, 2);
    }

    virtual TensorVec compute(OpContext &ctx, const TensorVec &inputs) {
        size_t n = inputs[0]->desc().numel();
        TensorPtr output = empty(inputs[0]->desc().dtype(), inputs[0]->desc().shape_vec());

#define BINARY_COMPUTE_DTYPE(dtype) kernel_<DTypeName::dtype>(ctx, inputs, output)
NCG_DTYPE_SWITCH_ALL(inputs[0]->desc().dtype(), BINARY_COMPUTE_DTYPE);
#undef BINARY_COMPUTE_DTYPE

        return {output};
    }

private:
    template <DTypeName DT>
    void kernel_(OpContext &ctx, const TensorVec &inputs, TensorPtr &output) {
        size_t n = inputs[0]->desc().numel();
        auto kernel = OpKernel();
        auto a = inputs[0]->as<DT>();
        auto b = inputs[1]->as<DT>();
        auto c = output->as<DT>();
        for (ssize_t i = 0; i < n; ++i) {
            kernel.template compute<DT>(ctx, this, a->elat(i), b->elat(i), c->mutable_elat(i));
        }
    }
};

enum class UnaryOpKernelType : int {
    Neg,
    Sin,
    Cos,
    Tan,
    Log,
    Exp,
    Tanh,
    Sigmoid,
    Reciprocal,
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
            case UnaryOpKernelType::Neg: b = -a; break;
            case UnaryOpKernelType::Sin: b = std::sin(a); break;
            case UnaryOpKernelType::Cos: b = std::cos(a); break;
            case UnaryOpKernelType::Tan: b = std::tan(a); break;
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
            case UnaryOpKernelType::Reciprocal:
                if (a == 0) {
                    ctx.error(op) << "Division by zero";
                    break;
                }
                b = 1 / a;
                break;
        }
    }
};

enum class BinaryOpKernelType : int {
    Add,
    Sub,
    Mul,
    Div,
    Ge,
    Le,
    Geq,
    Leq,
    Eq,
    Neq,
    Pow,
    Min,
    Max
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
            case BinaryOpKernelType::Ge: c = a > b; break;
            case BinaryOpKernelType::Le: c = a < b; break;
            case BinaryOpKernelType::Geq: c = a >= b; break;
            case BinaryOpKernelType::Leq: c = a <= b; break;
            case BinaryOpKernelType::Eq: c = a == b; break;
            case BinaryOpKernelType::Neq: c = a != b; break;
            case BinaryOpKernelType::Pow: c = std::pow(a, b); break;
            case BinaryOpKernelType::Min: c = std::min(a, b); break;
            case BinaryOpKernelType::Max: c = std::max(a, b); break;
        }
    }
};

#define DEF_UNARY_ELEMWISE_OP(name) \
class Op##name : public OpUnaryElemwiseBase<UnaryOpKernel<UnaryOpKernelType::name>> { \
public: \
    NCG_OP_DEF_NAME(Op##name); \
}

DEF_UNARY_ELEMWISE_OP(Neg);
DEF_UNARY_ELEMWISE_OP(Sin);
DEF_UNARY_ELEMWISE_OP(Cos);
DEF_UNARY_ELEMWISE_OP(Tan);
DEF_UNARY_ELEMWISE_OP(Log);
DEF_UNARY_ELEMWISE_OP(Exp);
DEF_UNARY_ELEMWISE_OP(Tanh);
DEF_UNARY_ELEMWISE_OP(Sigmoid);
DEF_UNARY_ELEMWISE_OP(Reciprocal);

#undef DEF_UNARY_ELEMWISE_OP

#define DEF_BINARY_ELEMWISE_OP(name) \
class Op##name : public OpBinaryElemwiseBase<BinaryOpKernel<BinaryOpKernelType::name>> { \
public: \
    NCG_OP_DEF_NAME(Op##name); \
}

DEF_BINARY_ELEMWISE_OP(Add);
DEF_BINARY_ELEMWISE_OP(Sub);
DEF_BINARY_ELEMWISE_OP(Mul);
DEF_BINARY_ELEMWISE_OP(Div);
DEF_BINARY_ELEMWISE_OP(Ge);
DEF_BINARY_ELEMWISE_OP(Le);
DEF_BINARY_ELEMWISE_OP(Geq);
DEF_BINARY_ELEMWISE_OP(Leq);
DEF_BINARY_ELEMWISE_OP(Eq);
DEF_BINARY_ELEMWISE_OP(Neq);
DEF_BINARY_ELEMWISE_OP(Pow);
DEF_BINARY_ELEMWISE_OP(Min);
DEF_BINARY_ELEMWISE_OP(Max);

#undef DEF_BINARY_ELEMWISE_OP

} /* !namespace ncg */

#endif /* !CORE_OPS_ARITH_H */
