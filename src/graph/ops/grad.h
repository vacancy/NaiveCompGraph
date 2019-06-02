/*
 * grad.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#ifndef GRAPH_OPS_GRAD_H
#define GRAPH_OPS_GRAD_H

#include "core/tensor.h"
#include "graph/tensor.h"
#include "graph/op.h"

namespace ncg {

class GOpGradLoss : public GraphOp, public GraphSingleOutputOp {
public:
    NCG_DEF_GOPNAME(GOpGradLoss);

    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        if (inputs.size() != 1) {
            graph.error(this) << "GradLoss op takes only one input";
        }
        if (inputs[0]->desc().dim() != 0) {
            graph.error(this) << "GradLoss op support only scalar-typed inputs";
        }
    }

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        return {make_tensor(0, TensorDesc(inputs[0]->desc().dtype(), {}))};
    }

    virtual void forward(GraphForwardContext &ctx) const {
        TensorPtr loss = ctx.tensor(m_inputs[0]);
        TensorPtr grad = scalar(loss->desc().dtype(), 1);
        ctx.set_tensor(m_outputs[0], grad);
    }

    NCG_DEF_GOP_NO_GRAD_INLINE;
};

} /* !namespace ncg */

#endif /* !GRAPH_OPS_GRAD_H */
