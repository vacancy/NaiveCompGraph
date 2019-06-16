/*
 * reduction.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#pragma once

#include "core/tensor.h"
#include "core/tensor_impl.h"
#include "core/op.h"
#include "core/ops/reduction.h"

#include "graph/op.h"

namespace ncg {

template <typename OpClass>
class GOpReduceBase : public GraphOpWrapper<OpClass>, public GraphSingleOutputOp {
public:
    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        const auto &desc = this->template desc<OpReduceDesc>();
        NCG_OP_CHECK_NR_INPUTS(graph, inputs, 1);
        NCG_OP_CHECK_INPUT_DIM_GEQ(graph, inputs, 0, desc.axis);
    }
};

template <typename OpClass>
class GOpReduceType1Base : public GOpReduceBase<OpClass> {
public:
    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        const auto &desc = this->template desc<OpReduceDesc>();
        auto output_shape = inputs[0]->desc().shape_vec();
        if (desc.keepdims) {
            output_shape[desc.axis] = 1;
        } else {
            output_shape.erase(output_shape.begin() + desc.axis);
        }

        return {
            this->make_tensor(0, TensorDesc(inputs[0]->desc().dtype(), output_shape)),
            this->make_tensor(1, TensorDesc(DTypeName::Int64, output_shape))
        };
    }
};

template <typename OpClass>
class GOpReduceType2Base : public GOpReduceBase<OpClass> {
public:
    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        const auto &desc = this->template desc<OpReduceDesc>();
        auto output_shape = inputs[0]->desc().shape_vec();
        if (desc.keepdims) {
            output_shape[desc.axis] = 1;
        } else {
            output_shape.erase(output_shape.begin() + desc.axis);
        }

        return {this->make_tensor(0, TensorDesc(inputs[0]->desc().dtype(), output_shape))};
    }
};

#define DEF_GOP_REDUCE(name, type_id) \
class GOpReduce##name : public GOpReduceType##type_id##Base<OpReduce##name> { \
public: \
    NCG_GOP_DEF_NAME(GOp##name); \
    virtual void backward(Graph &graph, GTensorPtr loss); \
}

DEF_GOP_REDUCE(Max, 1);
DEF_GOP_REDUCE(Min, 1);
DEF_GOP_REDUCE(Sum, 1);
DEF_GOP_REDUCE(Mean, 1);

#undef DEF_GOP_REDUCE

} /* !namespace ncg */

