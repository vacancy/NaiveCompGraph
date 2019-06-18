/*
 * update.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#pragma once

#include "graph/tensor.h"
#include "graph/op.h"

namespace ncg {

class GOpAssign : public GraphOp, public GraphSingleOutputOp {
public:
    NCG_GOP_DEF_NAME(GOpAssign);

    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        NCG_OP_CHECK_NR_INPUTS(graph, inputs, 2);
        NCG_OP_CHECK_COMPATIBLE_DTYPE(graph, inputs);
        NCG_OP_CHECK_COMPATIBLE_SHAPE(graph, inputs);

        auto var_op = inputs[0]->template owner_op<GOpVariable>();
        if (var_op == nullptr) {
            graph.error(this) << "The first input to " << this->op_name() << " must be a variable.";
        }
    }

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        return {make_tensor(0, TensorDesc(inputs[0]->desc().dtype(), {}))};
    }

    virtual void forward(GraphForwardContext &ctx) const {
        auto variable = ctx.tensor(m_inputs[0]);
        auto new_value = ctx.tensor(m_inputs[1]);
        ctx.set_tensor(m_outputs[0], new_value);
    }

    virtual void forward_hook_post(GraphForwardContext &ctx) const {
        auto new_variable = ctx.tensor(m_inputs[1]);
        ctx.session().set_shared_tensor(m_inputs[0], new_variable);
    }

    NCG_GOP_DEF_NO_GRAD_INLINE;
};

} /* !namespace ncg */

