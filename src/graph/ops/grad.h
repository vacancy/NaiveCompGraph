/*
 * grad.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#pragma once

#include "core/tensor_impl.h"
#include "graph/tensor.h"
#include "graph/op.h"

namespace ncg {

class GOpGradLoss : public GraphOp, public GraphSingleOutputOp {
public:
    NCG_GOP_DEF_NAME(GOpGradLoss);

    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        NCG_OP_CHECK_NR_INPUTS(graph, inputs, 1);
        NCG_OP_CHECK_INPUT_SCALAR(graph, inputs, 0);
    }

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        return {make_tensor(0, TensorDesc(inputs[0]->desc().dtype(), {}))};
    }

    virtual void forward(GraphForwardContext &ctx) const {
        auto loss = ctx.tensor(m_inputs[0]);
        auto grad = scalar(loss->desc().dtype(), 1);
        ctx.set_tensor(m_outputs[0], grad);
    }

    NCG_GOP_DEF_NO_GRAD_INLINE;
};

} /* !namespace ncg */

