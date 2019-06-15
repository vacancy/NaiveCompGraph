/*
 * graph_op.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#pragma once

#include "core/op.h"
#include "graph/tensor.h"
#include "graph/graph.h"

#include <string>
#include <memory>

namespace ncg {

class GraphOp {
public:
    GraphOp();
    virtual ~GraphOp(void) = default;

    virtual const char *op_name() const = 0;
    std::string name() const;
    std::string auto_name() const;
    void set_name(const std::string &name);

    template <typename DescT>
    const DescT &desc() const {
        auto p = dynamic_cast<DescT *>(m_desc.get());
        ncg_assert(p != nullptr);
        return *p;
    }
    const GTensorVec &inputs() const;
    const GTensorVec &outputs() const;

    const GTensorVec & operator () (Graph &graph, OpDescPtr desc, const GTensorVec &inputs);
    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) = 0;
    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) = 0;

    virtual void forward(GraphForwardContext &ctx) const = 0;
    virtual void backward(Graph &graph, GTensorPtr loss);

    GTensorPtr make_tensor(ssize_t index, const TensorDesc &desc);
    friend std::ostream & operator << (std::ostream &, const GraphOp &);

protected:
    std::string m_name;
    bool m_name_initialized;

    OpDescPtr m_desc;
    GTensorVec m_inputs;
    GTensorVec m_outputs;
    bool m_initialized;
};

#define NCG_GOP_DEF_NAME(op_name_) virtual const char *op_name(void) const { return #op_name_; }

#define NCG_GOP_DEF_NO_GRAD(op_name_) void op_name_::backward(Graph &graph, GTensorPtr loss) { \
    for (auto &tensor : m_inputs) { tensor->set_grad(graph, loss, nullptr); } \
}

#define NCG_GOP_DEF_NO_GRAD_INLINE virtual void backward(Graph &graph, GTensorPtr loss) { \
    for (auto &tensor : m_inputs) { tensor->set_grad(graph, loss, nullptr); } \
}

class GraphSingleOutputOp {
    // Pass
};

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

} /* !namespace ncg */

