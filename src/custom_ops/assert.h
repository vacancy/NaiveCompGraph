/*
 * assert.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#pragma once

#include "core/tensor_impl.h"
#include "graph/op.h"

namespace ncg {

class OpAssert : public Op {
    NCG_OP_DEF_NAME(OpAssert);

    virtual void check_inputs(OpContext &ctx, const TensorVec &inputs) {
        NCG_OP_CHECK_NR_INPUTS(ctx, inputs, 1);
        NCG_OP_CHECK_INPUT_SCALAR(ctx, inputs, 0);
    }

    virtual TensorVec compute(OpContext &ctx, const TensorVec &inputs) {
        bool rv = true;

#define ASSERT_COMPUTE_DTYPE(dtype_name) rv = compute_inner_<DTypeName::dtype_name>(ctx, inputs[0])
NCG_DTYPE_SWITCH_ALL(inputs[0]->desc().dtype(), ASSERT_COMPUTE_DTYPE);
#undef ASSERT_COMPUTE_DTYPE

        if (!rv) {
            ctx.error(this) << "Assertion failed";
        }

        return {inputs[0]};
    }

private:
    template <DTypeName DT>
    bool compute_inner_(OpContext &ctx, const TensorPtr &input) {
        auto a = input->as<DT>();
        return a->elat(0) > 0;
    }
};

class GOpAssert : public GraphOpWrapper<OpAssert>, public GraphSingleOutputOp {
public:
    NCG_GOP_DEF_NAME(GOpAssert);

    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        NCG_OP_CHECK_NR_INPUTS(graph, inputs, 1);
        NCG_OP_CHECK_INPUT_SCALAR(graph, inputs, 0);
    }

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        return {make_tensor(0, inputs[0]->desc())};
    }

    virtual void backward(Graph &graph, GTensorPtr loss) {
        auto output_grad = m_outputs[0]->grad(loss);
        m_inputs[0]->set_grad(graph, loss, output_grad);
    }
};

} /* !namespace ncg */

