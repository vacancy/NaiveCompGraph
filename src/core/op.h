/*
 * op.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#pragma once

#include "core/common.h"
#include "core/tensor.h"

#include <string>
#include <sstream>
#include <iostream>

namespace ncg {

class OpContext;

class OpDesc {
public:
    virtual ~OpDesc() = default;
};

typedef std::shared_ptr<OpDesc> OpDescPtr;

class Op {
public:
    Op() : m_desc() {}
    virtual ~Op() = default;

    TensorVec execute(OpContext &ctx, const TensorVec &inputs);
    virtual void check_inputs(OpContext &ctx, const TensorVec &inputs) = 0;
    virtual TensorVec compute(OpContext &ctx, const TensorVec &inputs) = 0;
    virtual const char *op_name() const = 0;

    template <typename DescT>
    const DescT &desc() const {
        auto p = dynamic_cast<DescT *>(m_desc.get());
        ncg_assert(p != nullptr);
        return *p;
    }

    void set_desc(OpDescPtr);

    friend std::ostream & operator << (std::ostream &, const Op &);

protected:
    OpDescPtr m_desc;
};

#define NCG_OP_DEF_NAME(op_name_) virtual const char *op_name() const { return #op_name_; }

#define NCG_OP_CHECK_CTX_CLEAN(ctx) if (ctx.is_error()) return ;

#define NCG_OP_CHECK_NR_INPUTS(ctx, inputs, n) do { \
    if (inputs.size() != n) { \
        ctx.error(this) << this->op_name() << " requires " << n << " input(s), but got " << inputs.size() << " input(s)."; \
        return; \
    } \
} while (0)

#define NCG_OP_CHECK_NR_INPUTS2(ctx, inputs, n1, n2) do { \
    if (inputs.size() != n1 && inputs.size() != n2) { \
        ctx.error(this) << this->op_name() << " requires " << n1 << " or " << n2 << " input(s), but got " << inputs.size() << " input(s)."; \
        return; \
    } \
} while (0)

#define NCG_OP_CHECK_EMPTY_INPUTS(ctx, inputs) NCG_OP_CHECK_NR_INPUTS(ctx, inputs, 0)

#define NCG_OP_CHECK_NONEMPTY_INPUTS(ctx, inputs) do { \
    if (inputs.size() == 0) { \
        ctx.error(this) << this->op_name() << " requires at least one input, but got zero."; \
        return; \
    } \
} while (0)

#define NCG_OP_CHECK_COMPATIBLE_DTYPE(ctx, inputs) do { \
    if (inputs.size() > 0) { \
        for (ssize_t i = 1; i < inputs.size(); ++i) { \
            if (inputs[i]->desc().dtype() != inputs[0]->desc().dtype()) { \
                auto &err_stream = ctx.error(this) << this->op_name() << " requires all inputs to have the same dtype, but got: "; \
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
                auto &err_stream = ctx.error(this) << this->op_name() << " requires all inputs to have the same dimension, but got: "; \
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
                auto &err_stream = ctx.error(this) << this->op_name() << " requires all inputs to have compatible shapes, but got: "; \
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
                auto &err_stream = ctx.error(this) << this->op_name() << " requires all inputs to have broadcastable shapes, but got: "; \
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
        ctx.error(this) << this->op_name() << " requires the input " << (idx_value + 1) << " to have dtype " << #dtype_name << ", but got " << get_dtype_name(inputs[idx_value]->desc().dtype()) << "."; \
        return; \
    } \
} while (0)

#define NCG_OP_CHECK_INPUT_DTYPE_INT(ctx, inputs, idx) do { \
    auto idx_value = (idx); \
    if (inputs[idx_value]->desc().dtype() != DTypeName::Int32 && inputs[idx_value]->desc().dtype() != DTypeName::Int64) { \
        ctx.error(this) << this->op_name() << " requires the input " << (idx_value + 1) << " to have dtype Int32 or Int64, but got " << get_dtype_name(inputs[idx_value]->desc().dtype()) << "."; \
        return; \
    } \
} while (0)

#define NCG_OP_CHECK_INPUT_DTYPE_FLOAT(ctx, inputs, idx) do { \
    auto idx_value = (idx); \
    if (inputs[idx_value]->desc().dtype() != DTypeName::Float32 && inputs[idx_value]->desc().dtype() != DTypeName::Float64) { \
        ctx.error(this) << this->op_name() << " requires the input " << (idx_value + 1) << " to have dtype Float32 or Float64, but got " << get_dtype_name(inputs[idx_value]->desc().dtype()) << "."; \
        return; \
    } \
} while (0)

#define NCG_OP_CHECK_INPUT_DIM(ctx, inputs, idx, dim_expr) do { \
    auto idx_value = (idx); \
    auto dim_value = (dim_expr); \
    if (inputs[idx_value]->desc().dim() != dim_value) { \
        ctx.error(this) << this->op_name() << " requires the input " << (idx_value + 1) << " to have dimension " << dim_value << ", but got " << inputs[idx_value]->desc().dim() << "."; \
        return; \
    } \
} while (0)

#define NCG_OP_CHECK_INPUT_DIM_GEQ(ctx, inputs, idx, dim_expr) do { \
    auto idx_value = (idx); \
    auto dim_value = (dim_expr); \
    if (inputs[idx_value]->desc().dim() < dim_value) { \
        ctx.error(this) << this->op_name() << " requires the input " << (idx_value + 1) << " to have a dimension of at least " << dim_value << ", but got " << inputs[idx_value]->desc().dim() << "."; \
        return; \
    } \
} while (0)

#define NCG_OP_CHECK_INPUT_SCALAR(ctx, inputs, idx) do { \
    auto idx_value = (idx); \
    if (inputs[idx_value]->desc().dim() != 0) { \
        ctx.error(this) << this->op_name() << " requires the input " << (idx_value + 1) << " to be a scalar, but got a " << inputs[idx_value]->desc().dim() << " dimensional input."; \
        return; \
    } \
} while (0)

#define NCG_OP_CHECK_INPUT_VECTOR(ctx, inputs, idx) do { \
    auto idx_value = (idx); \
    if (inputs[idx_value]->desc().dim() != 1) { \
        ctx.error(this) << this->op_name() << " requires the input " << (idx_value + 1) << " to be a vector, but got a " << inputs[idx_value]->desc().dim() << " dimensional input."; \
        return; \
    } \
} while (0)

#define NCG_OP_CHECK_INPUT_SCALAR_VECTOR(ctx, inputs, idx) do { \
    auto idx_value = (idx); \
    if (inputs[idx_value]->desc().dim() > 1) { \
        ctx.error(this) << this->op_name() << " requires the input " << (idx_value + 1) << " to be a scalar or a vector, but got a " << inputs[idx_value]->desc().dim() << " dimensional input."; \
        return; \
    } \
} while (0)

class OpContext : public RuntimeContext {
public:
    OpContext() = default;
    virtual ~OpContext() = default;

    std::ostringstream &error(const Op *);
};

} /* !namespace ncg */

