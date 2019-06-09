/*
 * op_common.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#ifndef CORE_OPS_COMMON_H
#define CORE_OPS_COMMON_H

#include "core/op.h"

namespace ncg {

#define NCG_OP_CHECK_CTX_CLEAN(ctx) if (ctx.is_error()) return ;

#define NCG_OP_CHECK_NR_INPUTS(ctx, inputs, n) do { \
    if (inputs.size() != n) { \
        ctx.error(this) << op_name() << " requires " << n << " input(s), but got " << inputs.size() << " input(s)."; \
        return; \
    } \
} while (0)

#define NCG_OP_CHECK_NONEMPTY_INPUTS(ctx, inputs) do { \
    if (inputs.size() == 0) { \
        ctx.error(this) << op_name() << " requires at least one input, but got zero"; \
        return; \
    } \
} while (0)

#define NCG_OP_CHECK_COMPATIBLE_DTYPE(ctx, inputs) do { \
    if (inputs.size() > 0) { \
        for (ssize_t i = 1; i < inputs.size(); ++i) { \
            if (inputs[i]->desc().dtype() != inputs[0]->desc().dtype()) { \
                auto &err_stream = ctx.error(this) << op_name() << " requires all inputs have the same dtype, but got: "; \
                for (ssize_t j = 0; j < inputs.size(); ++j) { \
                    if (j != 0) err_stream << ", "; \
                    err_stream << get_dtype_name(inputs[j]->desc().dtype()); \
                } \
                err_stream << "."; \
                return; \
            } \
        } \
    } \
} while (0)

#define NCG_OP_CHECK_COMPATIBLE_DIM(ctx, inputs) do { \
    if (inputs.size() > 0) { \
        for (ssize_t i = 1; i < inputs.size(); ++i) { \
            if (inputs[i]->desc().dim() != inputs[0]->desc().dim()) { \
                auto &err_stream = ctx.error(this) << op_name() << " requires all inputs have the same dimension, but got: "; \
                for (ssize_t j = 0; j < inputs.size(); ++j) { \
                    if (j != 0) err_stream << ", "; \
                    err_stream << inputs[j]->desc().dim(); \
                } \
                err_stream << "."; \
                return; \
            } \
        } \
    } \
} while (0)

#define NCG_OP_CHECK_COMPATIBLE_SHAPE_PRINT(ctx, inputs, err_stream) do { \
    for (ssize_t j = 0; j < inputs.size(); ++j) { \
        if (j != 0) err_stream << ", "; \
        err_stream << inputs[j]->desc().shape_vec(); \
    } \
} while(0)

#define NCG_OP_CHECK_COMPATIBLE_SHAPE(ctx, inputs) do { \
    if (inputs.size() > 0) { \
        for (ssize_t i = 1; i < inputs.size(); ++i) { \
            if (!inputs[i]->desc().is_compatible(inputs[0]->desc())) { \
                auto &err_stream = ctx.error(this) << op_name() << " requires all inputs have compatible shapes, but got: "; \
                NCG_OP_CHECK_COMPATIBLE_SHAPE_PRINT(ctx, inputs, err_stream); \
                err_stream << "."; \
                return; \
            } \
        } \
    } \
} while (0)

#define NCG_OP_CHECK_BROADCASTABLE_SHAPE(ctx, inputs) do { \
    if (inputs.size() > 0) { \
        for (ssize_t i = 1; i < inputs.size(); ++i) { \
            if (!inputs[i]->desc().is_compatible(inputs[0]->desc(), true)) { \
                auto &err_stream = ctx.error(this) << op_name() << " requires all inputs have broadcastable shapes, but got: "; \
                NCG_OP_CHECK_COMPATIBLE_SHAPE_PRINT(ctx, inputs, err_stream); \
                err_stream << "."; \
                return; \
            } \
        } \
    } \
} while (0)

#define NCG_OP_CHECK_INPUT_DTYPE(ctx, inputs, idx, dtype_name) do { \
    auto idx_value = (idx); \
    if (inputs[idx_value]->desc().dtype() != DTypeName::dtype_name) { \
        ctx.error(this) << op_name() << " requires the input " << (idx_value + 1) << " has dtype " << #dtype_name << ", but got " << get_dtype_name(inputs[idx_value]->desc().dtype()) << "."; \
        return; \
    } \
} while (0)

#define NCG_OP_CHECK_INPUT_DTYPE_INT(ctx, inputs, idx) do { \
    auto idx_value = (idx); \
    if (inputs[idx_value]->desc().dtype() != DTypeName::Int32 && inputs[idx_value]->desc().dtype() != DTypeName::Int64) { \
        ctx.error(this) << op_name() << " requires the input " << (idx_value + 1) << " has dtype Int32 or Int64, but got " << get_dtype_name(inputs[idx_value]->desc().dtype()) << "."; \
        return; \
    } \
} while (0)

#define NCG_OP_CHECK_INPUT_DTYPE_FLOAT(ctx, inputs, idx) do { \
    auto idx_value = (idx); \
    if (inputs[idx_value]->desc().dtype() != DTypeName::Float32 && inputs[idx_value]->desc().dtype() != DTypeName::Float64) { \
        ctx.error(this) << op_name() << " requires the input " << (idx_value + 1) << " has dtype Float32 or Float64, but got " << get_dtype_name(inputs[idx_value]->desc().dtype()) << "."; \
        return; \
    } \
} while (0)

#define NCG_OP_CHECK_INPUT_DIM(ctx, inputs, idx, dim_expr) do { \
    auto idx_value = (idx); \
    auto dim_value = (dim_expr); \
    if (inputs[idx_value]->desc().dim() != dim_value) { \
        ctx.error(this) << op_name() << " requires the input " << (idx_value + 1) << " has dimension " << dim_value << ", but got " << inputs[idx_value]->desc().dim() << "."; \
        return; \
    } \
} while (0)


class ElemWiseOp : public Op {
public:
    virtual void check_inputs(OpContext &ctx, const TensorVec &inputs) {
        NCG_OP_CHECK_NONEMPTY_INPUTS(ctx, inputs);
        NCG_OP_CHECK_COMPATIBLE_DTYPE(ctx, inputs);
        NCG_OP_CHECK_COMPATIBLE_SHAPE(ctx, inputs);
    }
};

template <typename OpKernel>
class UnaryElemWiseOp : public ElemWiseOp {
public:
    virtual void check_inputs(OpContext &ctx, const TensorVec &inputs) {
        ElemWiseOp::check_inputs(ctx, inputs);
        NCG_OP_CHECK_CTX_CLEAN(ctx);
        NCG_OP_CHECK_NR_INPUTS(ctx, inputs, 1);
    }

    virtual TensorVec compute(OpContext &ctx, const TensorVec &inputs) {
        TensorPtr output = empty(inputs[0]->desc().dtype(), inputs[0]->desc().shape_vec());

#define UNARY_COMPUTE_DTYPE(dtype) compute_inner_<DTypeName::dtype>(ctx, inputs, output)
NCG_SWITCH_DTYPE_ALL(inputs[0]->desc().dtype(), UNARY_COMPUTE_DTYPE);
#undef UNARY_COMPUTE_DTYPE

        return {output};
    }

private:
    template <DTypeName DT>
    void compute_inner_(OpContext &ctx, const TensorVec &inputs, TensorPtr &output) {
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
class BinaryElemWiseOp : public ElemWiseOp {
public:
    virtual void check_inputs(OpContext &ctx, const TensorVec &inputs) {
        ElemWiseOp::check_inputs(ctx, inputs);
        NCG_OP_CHECK_CTX_CLEAN(ctx);
        NCG_OP_CHECK_NR_INPUTS(ctx, inputs, 2);
    }

    virtual TensorVec compute(OpContext &ctx, const TensorVec &inputs) {
        size_t n = inputs[0]->desc().numel();
        TensorPtr output = empty(inputs[0]->desc().dtype(), inputs[0]->desc().shape_vec());

#define BINARY_COMPUTE_DTYPE(dtype) compute_inner_<DTypeName::dtype>(ctx, inputs, output)
NCG_SWITCH_DTYPE_ALL(inputs[0]->desc().dtype(), BINARY_COMPUTE_DTYPE);
#undef BINARY_COMPUTE_DTYPE

        return {output};
    }

private:
    template <DTypeName DT>
    void compute_inner_(OpContext &ctx, const TensorVec &inputs, TensorPtr &output) {
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

} /* !namespace ncg */

#endif /* !CORE_OPS_COMMON_H */
