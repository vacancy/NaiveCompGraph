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

    virtual void backward(Graph &graph, GTensorPtr loss) {
        auto output_grad = m_outputs[0]->grad(loss);
        m_inputs[0]->set_grad(graph, loss, output_grad);
        m_inputs[1]->set_grad(graph, loss, nullptr);
    }
};

} /* !namespace ncg */

#endif /* !CUSTOM_OPS_BIND_H */
