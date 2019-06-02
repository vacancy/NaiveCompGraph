/*
 * bind.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#ifndef CUSTOM_OPS_BIND_H
#define CUSTOM_OPS_BIND_H

namespace ncg {

class GOpBind : public GraphOp, public GraphSingleOutputOp {
public:
    NCG_DEF_GOPNAME(GOpBind);

    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        if (inputs.size() != 2) {
            graph.error(this) << "Bind op takes two inputs";
        }
    }
    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        return {make_tensor(0, inputs[0]->desc())};
    }

    virtual void forward(GraphForwardContext &ctx) const {
        TensorPtr tensor = ctx.tensor(m_inputs[0]);
        ctx.set_tensor(m_outputs[0], tensor);
    }

    NCG_DEF_GOP_NO_GRAD_INLINE;
};

} /* !namespace ncg */

#endif /* !CUSTOM_OPS_BIND_H */
