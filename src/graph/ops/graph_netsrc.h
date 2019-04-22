/*
 * graph_netsrc.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#ifndef GRAPH_NETSRC_H
#define GRAPH_NETSRC_H

#include "graph/graph_op.h"
#include <vector>

namespace ncg {

class GraphNetSrcOpDesc : public OpDesc {
public:
    GraphNetSrcOpDesc() : desc() {}
    GraphNetSrcOpDesc(DTypeName dtype, const std::vector<size_t> &shape) : desc(dtype, shape) {}
    virtual ~GraphNetSrcOpDesc() = default;

    TensorDesc desc;
};

class GraphNetSrcOp : public GraphOp, public GraphSingleOutputOp {
public:
    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        if (inputs.size() > 0) {
            graph.error(this) << "netsrc ops do not take any inputs.";
        }
    }
    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        return {make_tensor(0, this->template desc<GraphNetSrcOpDesc>().desc)};
    }
};

class GOpPlaceholder : public GraphNetSrcOp {
public:
    NCG_DEF_GOPNAME(GOpPlaceholder);

    virtual void forward(GraphForwardContext &ctx) const {
        TensorPtr tensor = ctx.feed_dict(name());
        if (!tensor) {
            ctx.error(this) << "value not found in the feed dict.";
        }
        ctx.set_tensor(m_outputs[0], tensor);
    }
};

class GOpConstant : public GraphNetSrcOp {
public:
    NCG_DEF_GOPNAME(GOpConstant);
};

class GOpVariable : public GraphNetSrcOp {
public:
    NCG_DEF_GOPNAME(GOpVariable);
};

} /* !namespace ncg */

#endif /* !GRAPH_NETSRC_H */
