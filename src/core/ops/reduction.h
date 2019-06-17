/*
 * reduction.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#pragma once

#include "core/op.h"

namespace ncg {

enum class ReduceType1 : int {
    Max,
    Min
};

enum class ReduceType2 : int {
    Sum,
    Mean,
    Prod
};

class OpReduceDesc : public OpDesc {
public:
    OpReduceDesc(ssize_t axis = 0, bool keepdims = false) : axis(axis), keepdims(keepdims) {}
    virtual ~OpReduceDesc() = default;

    ssize_t axis;
    bool keepdims;
};

class OpReduceBase : public Op {
public:
    virtual void check_inputs(OpContext &ctx, const TensorVec &inputs) {
        const auto &desc = this->template desc<OpReduceDesc>();
        auto axis = desc.axis;
        if (axis < 0) axis += inputs[0]->desc().dim();

        NCG_OP_CHECK_NR_INPUTS(ctx, inputs, 1);
        NCG_OP_CHECK_INPUT_DIM_GEQ(ctx, inputs, 0, axis);
    }
};

template <ReduceType1 ReduceType>
class OpReduceType1Base : public OpReduceBase {
public:
    virtual TensorVec compute(OpContext &ctx, const TensorVec &inputs) {
        const auto input = inputs[0];
        const auto &desc = this->template desc<OpReduceDesc>();
        auto axis = desc.axis;
        if (axis < 0) axis += inputs[0]->desc().dim();

        TensorVec outputs;

#define REDUCE_DTYPE_CASE(dtype_name) outputs = kernel_(ctx, input->template as<DTypeName::dtype_name>(), axis, desc.keepdims)
NCG_DTYPE_SWITCH_ALL(input->desc().dtype(), REDUCE_DTYPE_CASE);
#undef REDUCE_DTYPE_CASE

        return outputs;
    }

private:
    template <DTypeName DT>
    TensorVec kernel_(OpContext &ctx, TensorImpl<DT> *input, ssize_t axis, bool keepdims) {
        auto output_shape = input->desc().shape_vec();
        if (keepdims) {
            output_shape[axis] = 1;
        } else {
            output_shape.erase(output_shape.begin() + axis);
        }

        TensorPtr output_ptr, indices_ptr;
        if (ReduceType == ReduceType1::Min) {
            output_ptr = fill(DT, output_shape, std::numeric_limits<typename DType<DT>::cctype>::max());
        } else {
            output_ptr = fill(DT, output_shape, std::numeric_limits<typename DType<DT>::cctype>::lowest());
        }
        indices_ptr = empty(DTypeName::Int64, output_shape);

        auto output = output_ptr->template as<DT>();
        auto indices = indices_ptr->template as<DTypeName::Int64>();

        auto input_data_ptr = input->data_ptr();
        bool input_con = input->desc().is_contiguous();
        auto output_data_ptr = output->mutable_data_ptr();
        auto indices_data_ptr = indices->mutable_data_ptr();

        const auto &input_default_stride = input->desc().get_default_stride();
        const auto &output_default_stride = output->desc().get_default_stride();
        for (ssize_t i = 0; i < input->desc().numel(); ++i) {
            ssize_t j1, j2, j3;
            if (axis != 0) {
                j1 = i / input_default_stride[axis - 1];
                j2 = (i % input_default_stride[axis - 1]) / input_default_stride[axis];
                j3 = i % input_default_stride[axis];
            } else {
                j1 = 0;
                j2 = i / input_default_stride[axis];
                j3 = i % input_default_stride[axis];
            }

            ssize_t j;
            if (axis != 0) {
                j = j1 * output_default_stride[axis - 1] + j3;
            } else {
                j = j3;
            }

            const auto input_val = input_con ? input_data_ptr[i] : input->elat(i);
            if (ReduceType == ReduceType1::Min) {
                if (input_val < output_data_ptr[j]) {
                    output_data_ptr[j] = input_val;
                    indices_data_ptr[j] = static_cast<DType<DTypeName::Int64>::cctype>(j2);
                }
            } else {
                if (input->elat(i) > output_data_ptr[j]) {
                    output_data_ptr[j] = input_val;
                    indices_data_ptr[j] = static_cast<DType<DTypeName::Int64>::cctype>(j2);
                }
            }
        }

        return {output_ptr, indices_ptr};
    }
};

class OpReduceMax : public OpReduceType1Base<ReduceType1::Max> {
public:
    NCG_OP_DEF_NAME(OpReduceMax);
};

class OpReduceMin : public OpReduceType1Base<ReduceType1::Min> {
public:
    NCG_OP_DEF_NAME(OpReduceMin);
};

template <ReduceType2 ReduceType>
class OpReduceType2Base : public OpReduceBase {
public:
    virtual TensorVec compute(OpContext &ctx, const TensorVec &inputs) {
        const auto input = inputs[0];
        const auto &desc = this->template desc<OpReduceDesc>();
        auto axis = desc.axis;
        if (axis < 0) axis += inputs[0]->desc().dim();

        TensorVec outputs;

#define REDUCE_DTYPE_CASE(dtype_name) outputs = kernel_(ctx, input->template as<DTypeName::dtype_name>(), axis, desc.keepdims)
NCG_DTYPE_SWITCH_ALL(input->desc().dtype(), REDUCE_DTYPE_CASE);
#undef REDUCE_DTYPE_CASE

        return outputs;
    }

private:
    template <DTypeName DT>
    TensorVec kernel_(OpContext &ctx, TensorImpl<DT> *input, ssize_t axis, bool keepdims) {
        auto output_shape = input->desc().shape_vec();
        auto axis_size = static_cast<typename DType<DT>::cctype>(output_shape[axis]);
        if (keepdims) {
            output_shape[axis] = 1;
        } else {
            output_shape.erase(output_shape.begin() + axis);
        }

        TensorPtr output_ptr;
        if (ReduceType == ReduceType2::Sum || ReduceType == ReduceType2::Mean) {
            output_ptr = fill(DT, output_shape, 0);
        } else {
            output_ptr = fill(DT, output_shape, 1);
        }
        auto output = output_ptr->template as<DT>();

        auto input_data_ptr = input->data_ptr();
        bool input_con = input->desc().is_contiguous();
        auto output_data_ptr = output->mutable_data_ptr();

        const auto &input_default_stride = input->desc().get_default_stride();
        const auto &output_default_stride = output->desc().get_default_stride();
        for (ssize_t i = 0; i < input->desc().numel(); ++i) {
            ssize_t j1, j2, j3;
            if (axis != 0) {
                j1 = i / input_default_stride[axis - 1];
                j2 = (i % input_default_stride[axis - 1]) / input_default_stride[axis];
                j3 = i % input_default_stride[axis];
            } else {
                j1 = 0;
                j2 = i / input_default_stride[axis];
                j3 = i % input_default_stride[axis];
            }

            ssize_t j;
            if (axis != 0) {
                j = j1 * output_default_stride[axis - 1] + j3;
            } else {
                j = j3;
            }

            const auto input_val = input_con ? input_data_ptr[i] : input->elat(i);
            if (ReduceType == ReduceType2::Sum) {
                output_data_ptr[j] += input_val;
            } else if (ReduceType == ReduceType2::Mean) {
                output_data_ptr[j] += input_val / axis_size;
            } else if (ReduceType == ReduceType2::Prod) {
                output_data_ptr[j] *= input_val;
            }
        }

        return {output_ptr};
    }
};

class OpReduceSum : public OpReduceType2Base<ReduceType2::Sum> {
public:
    NCG_OP_DEF_NAME(OpReduceSum);
};

class OpReduceMean : public OpReduceType2Base<ReduceType2::Mean> {
public:
    NCG_OP_DEF_NAME(OpReduceMean);
};

class OpReduceProd : public OpReduceType2Base<ReduceType2::Prod> {
public:
    NCG_OP_DEF_NAME(OpReduceProd);
};

} /* !namespace ncg */

