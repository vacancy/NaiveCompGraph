/*
 * graph_op_common.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#ifndef GRAPH_OPS_OP_COMMON_H
#define GRAPH_OPS_OP_COMMON_H

#include "graph/op.h"
#include "graph/graph.h"
#include <memory>

namespace ncg {

template <typename OpClass>
class GraphOpWrapper : public GraphOp {
public:
    virtual void forward(GraphForwardContext &ctx) const {
        TensorVec inputs;
        for (const auto &gtensor : m_inputs) {
            inputs.push_back(ctx.tensor(gtensor));
        }
        auto op = std::make_unique<OpClass>();
        op->set_desc(m_desc);
        TensorVec outputs = op->execute(ctx, inputs);
        ncg_assert(m_outputs.size() == outputs.size());
        for (ssize_t i = 0; i < m_outputs.size(); ++i) {
            ctx.set_tensor(m_outputs[i], outputs[i]);
        }
    }
};

template <typename OpClass>
class GraphElemWiseOp : public GraphOpWrapper<OpClass> {
public:
    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        if (inputs.size() == 0) {
            return ;
        }
        for (ssize_t i = 0; i < inputs.size(); ++i) {
            if (!inputs[0]->desc().is_compatible(inputs[i]->desc())) {
                graph.error(this) << "incompatible inputs. inputs[0] = " << inputs[0]->desc() << "; inputs[" << i << "]=" << inputs[i]->desc() << ".";
            }
        }
    }
};

template <typename OpClass>
class GraphUnaryElemWiseOp : public GraphElemWiseOp<OpClass>, public GraphSingleOutputOp {
public:
    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        GraphElemWiseOp<OpClass>::check_inputs(graph, inputs);
        if (!graph.ok()) return;
        if (inputs.size() != 1) {
            graph.error(this) << "requires 1 input tensor.";
        }
    }

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        TensorDesc desc(inputs[0]->desc().dtype(), inputs[0]->desc().shape_vec());
        return {this->make_tensor(0, desc)};
    }

};

template <typename OpClass>
class GraphBinaryElemWiseOp : public GraphElemWiseOp<OpClass>, public GraphSingleOutputOp {
public:
    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        GraphElemWiseOp<OpClass>::check_inputs(graph, inputs);
        if (!graph.ok()) return;
        if (inputs.size() != 2) {
            graph.error(this) << "requires 2 input tensors.";
        }
    }

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        TensorDesc desc(inputs[0]->desc().dtype(), inputs[0]->desc().shape_vec());
        return {this->make_tensor(0, desc)};
    }
};

} /* !namespace ncg */

#endif /* !GRAPH_OPS_OP_COMMON_H */
