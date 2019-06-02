/*
 * graph_op.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#ifndef GRAPH_OP_H
#define GRAPH_OP_H

#include "core/op.h"
#include "graph/tensor.h"

#include <string>
#include <memory>

namespace ncg {

// From graph/graph.h
class Graph;
class GraphForwardContext;

class GraphOp {
public:
    GraphOp();
    virtual ~GraphOp(void) = default;

    virtual const char *op_name(void) const = 0;
    std::string name(void) const;
    std::string auto_name(void) const;
    void set_name(const std::string &name);

    template <typename DescT>
    DescT &desc() {
        return *(dynamic_cast<DescT *>(m_desc.get()));
    }
    template <typename DescT>
    const DescT &desc() const {
        return *(dynamic_cast<DescT *>(m_desc.get()));
    }
    const GTensorVec &inputs(void) const;
    const GTensorVec &outputs(void) const;

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

#define NCG_DEF_GOPNAME(op_name_) virtual const char *op_name(void) const { return #op_name_; }

#define NCG_DEF_GOP_NO_GRAD(op_name_) void op_name_::backward(Graph &graph, GTensorPtr loss) { \
    for (auto &tensor : m_inputs) { tensor->set_grad(graph, loss, nullptr); } \
}

#define NCG_DEF_GOP_NO_GRAD_INLINE virtual void backward(Graph &graph, GTensorPtr loss) { \
    for (auto &tensor : m_inputs) { tensor->set_grad(graph, loss, nullptr); } \
}


class GraphSingleOutputOp {
    // Pass
};

typedef std::shared_ptr<GraphOp> GOpPtr;

} /* !namespace ncg */

#endif /* !GRAPH_OP_H */
