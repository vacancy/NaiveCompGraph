/*
 * elemwise.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#pragma once

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

#define CAST_DTYPE_CASE(dtype_name) kernel_<DTypeName::dtype_name>(ctx, inputs, output)
NCG_DTYPE_SWITCH_ALL(inputs[0]->desc().dtype(), CAST_DTYPE_CASE);
#undef CAST_DTYPE_CASE

        return {output};
    }

private:
    template <DTypeName DT>
    void kernel_(OpContext &ctx, const TensorVec &inputs, TensorPtr &output) {
        const auto &desc = this->template desc<OpCastDesc>();
#define CAST_DTYPE_CASE(dtype_name) kernel_inner_<DT, DTypeName::dtype_name>(ctx, inputs, output)
NCG_DTYPE_SWITCH_ALL(desc.dtype, CAST_DTYPE_CASE);
#undef CAST_DTYPE_CASE
    }

    template <DTypeName DT, DTypeName ODT>
    void kernel_inner_(OpContext &ctx, const TensorVec &inputs, TensorPtr &output) {
        size_t n = inputs[0]->desc().numel();
        auto a = inputs[0]->as<DT>();
        auto b = output->as<ODT>();

        auto a_ptr = a->data_ptr();
        auto b_ptr = b->mutable_data_ptr();
        bool a_con = inputs[0]->desc().is_contiguous();

        if (a_con) {
            for (ssize_t i = 0; i < n; ++i) {
                b_ptr[i] = static_cast<typename DType<DT>::cctype>(a_ptr[i]);
            }
        } else {
        	for (ssize_t i = 0; i < n; ++i) {
            	b_ptr[i] = static_cast<typename DType<DT>::cctype>(a->elat(i));
            }
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

        auto a_ptr = a->data_ptr(), b_ptr = b->data_ptr(), c_ptr = c->data_ptr();
        auto d_ptr = d->mutable_data_ptr();
        bool a_con = a->desc().is_contiguous(), b_con = b->desc().is_contiguous(), c_con = c->desc().is_contiguous();

        for (ssize_t i = 0; i < n; ++i) {
            d_ptr[i] = (a_con ? a_ptr[i] : a->elat(i)) > 0 ? (b_con ? b_ptr[i] : b->elat(i)) : (c_con ? c_ptr[i] : c->elat(i));
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

template <UnaryOpKernelType OpType, DTypeName DT>
struct UnaryOpKernel {
    UnaryOpKernel(Op *self, OpContext &ctx) {
        if (OpType != UnaryOpKernelType::Neg) {
            if (DT != DTypeName::Float32 && DT != DTypeName::Float64) {
                ctx.error(self) << "Unary Op not implemented for non-float tensors.";
            }
        }
    }

    void compute(
        OpContext &ctx,
        Op *op,
        const typename DType<DT>::cctype &a,
        typename DType<DT>::cctype &b
    ) {
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

template <UnaryOpKernelType OpKernelType>
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
        auto kernel = UnaryOpKernel<OpKernelType, DT>(this, ctx);
        auto a = inputs[0]->as<DT>();
        auto b = output->as<DT>();

        auto a_ptr = a->data_ptr();
        auto b_ptr = b->mutable_data_ptr();
        bool a_con = a->desc().is_contiguous();

        if (a_con) {
            for (ssize_t i = 0; i < n; ++i) {
                kernel.compute(ctx, this, a_ptr[i], b_ptr[i]);
            }
        } else {
            for (ssize_t i = 0; i < n; ++i) {
                kernel.compute(ctx, this, a->elat(i), b_ptr[i]);
            }
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

template <BinaryOpKernelType OpType, DTypeName DT>
struct BinaryOpKernel {
    BinaryOpKernel(Op *self, OpContext &ctx) {}

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

template <BinaryOpKernelType OpKernelType>
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
        auto kernel = BinaryOpKernel<OpKernelType, DT>(this, ctx);
        auto a = inputs[0]->as<DT>();
        auto b = inputs[1]->as<DT>();
        auto c = output->as<DT>();

        auto a_ptr = a->data_ptr(), b_ptr = b->data_ptr();
        auto c_ptr = c->mutable_data_ptr();
        bool a_con = a->desc().is_contiguous(), b_con = b->desc().is_contiguous();
        bool a_sca = a->desc().is_scalar_broadcasted(), b_sca = b->desc().is_scalar_broadcasted();

#define BINARY_KERNEL_CASE(a_condition, b_condition, a_index, b_index) else if (a_condition && b_condition) { \
    for (ssize_t i = 0; i < n; ++i) { \
        kernel.compute(ctx, this, a_index, b_index, c_ptr[i]); \
    } \
}

        if (false) {}
        BINARY_KERNEL_CASE(a_con, b_con, a_ptr[i], b_ptr[i])
        BINARY_KERNEL_CASE(a_con, b_sca, a_ptr[i], b_ptr[0])
        BINARY_KERNEL_CASE(a_sca, b_con, a_ptr[0], b_ptr[i])
        BINARY_KERNEL_CASE(a_sca, b_sca, a_ptr[0], b_ptr[0])
        BINARY_KERNEL_CASE(true,  b_con, a->elat(i), b_ptr[i])
        BINARY_KERNEL_CASE(true,  b_sca, a->elat(i), b_ptr[0])
        BINARY_KERNEL_CASE(a_con, true,  a_ptr[i], b->elat(i))
        BINARY_KERNEL_CASE(a_sca, true,  a_ptr[0], b->elat(i))
        else {
            for (ssize_t i = 0; i < n; ++i) {
                kernel.compute(ctx, this, a->elat(i), b->elat(i), c_ptr[i]);
            }
        }
    }
};

#define DEF_UNARY_ELEMWISE_OP(name) \
class Op##name : public OpUnaryElemwiseBase<UnaryOpKernelType::name> { \
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
class Op##name : public OpBinaryElemwiseBase<BinaryOpKernelType::name> { \
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

