/*
 * graph_netsrc.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#ifndef GRAPH_OPS_NETSRC_H
#define GRAPH_OPS_NETSRC_H

#include "graph/op.h"
#include <vector>

namespace ncg {

class GraphNetSrcOp : public GraphOp, public GraphSingleOutputOp {
public:
    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        if (inputs.size() > 0) {
            graph.error(this) << "NetSrc ops do not take any inputs";
        }
    }

    virtual void backward(Graph &graph, GTensorPtr loss) {
        // NOP.
    }
};

class GOpPlaceholderDesc : public OpDesc {
public:
    GOpPlaceholderDesc() : desc() {}
    GOpPlaceholderDesc(DTypeName dtype, const shape_vec &shape) : desc(dtype, shape) {}
    virtual ~GOpPlaceholderDesc() = default;

    TensorDesc desc;
};

class GOpPlaceholder : public GraphNetSrcOp {
public:
    NCG_DEF_GOPNAME(GOpPlaceholder);

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        return {make_tensor(0, this->template desc<GOpPlaceholderDesc>().desc)};
    }
    virtual void forward(GraphForwardContext &ctx) const {
        TensorPtr tensor = ctx.feed_dict(name());
        if (!tensor) {
            ctx.error(this) << "Placeholder missing";
        }
        ctx.set_tensor(m_outputs[0], tensor);
    }
};

class GOpConstantDesc : public OpDesc {
public:
    GOpConstantDesc() : tensor() {}
    GOpConstantDesc(const TensorPtr &tensor) : tensor(tensor) {}
    virtual ~GOpConstantDesc() = default;

    TensorPtr tensor;
};

class GOpConstant : public GraphNetSrcOp {
public:
    NCG_DEF_GOPNAME(GOpConstant);

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        return {make_tensor(0, this->template desc<GOpConstantDesc>().tensor->desc())};
    }
    virtual void forward(GraphForwardContext &ctx) const {
        ctx.set_tensor(m_outputs[0], this->template desc<GOpConstantDesc>().tensor);
    }
};

class GOpVariableDesc : public OpDesc {
public:
    GOpVariableDesc() : tensor() {}
    GOpVariableDesc(const TensorPtr &tensor) : tensor(tensor) {}
    virtual ~GOpVariableDesc() = default;

    TensorPtr tensor;
};

class GOpVariable : public GraphNetSrcOp {
public:
    NCG_DEF_GOPNAME(GOpVariable);

    void set_value(const TensorPtr &tensor) {
        /* TODO: Check shape and dtype. */
        this->template desc<GOpVariableDesc>().tensor = tensor;
    }
    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        return {make_tensor(0, this->template desc<GOpVariableDesc>().tensor->desc())};
    }
    virtual void forward(GraphForwardContext &ctx) const {
        ctx.set_tensor(m_outputs[0], this->template desc<GOpVariableDesc>().tensor);
    }
};

class GOpZerosDesc : public OpDesc {
public:
    GOpZerosDesc() : desc() {}
    GOpZerosDesc(DTypeName dtype, const shape_vec &shape) : desc(dtype, shape) {}
    virtual ~GOpZerosDesc() = default;

    TensorDesc desc;
};

class GOpZeros: public GraphNetSrcOp {
public:
    NCG_DEF_GOPNAME(GOpZeros);

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        return {make_tensor(0, this->template desc<GOpZerosDesc>().desc)};
    }
    virtual void forward(GraphForwardContext &ctx) const {
        auto &desc = this->template desc<GOpZerosDesc>().desc;
        auto tensor = zeros(desc.dtype(), desc.shape_vec());
        ctx.set_tensor(m_outputs[0], tensor);
    }
};

class GOpOnesDesc : public OpDesc {
public:
    GOpOnesDesc() : desc() {}
    GOpOnesDesc(DTypeName dtype, const shape_vec &shape) : desc(dtype, shape) {}
    virtual ~GOpOnesDesc() = default;

    TensorDesc desc;
};

class GOpOnes: public GraphNetSrcOp {
public:
    NCG_DEF_GOPNAME(GOpZeros);

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        return {make_tensor(0, this->template desc<GOpOnesDesc>().desc)};
    }
    virtual void forward(GraphForwardContext &ctx) const {
        auto &desc = this->template desc<GOpOnesDesc>().desc;
        auto tensor = ones(desc.dtype(), desc.shape_vec());
        ctx.set_tensor(m_outputs[0], tensor);
    }
};

} /* !namespace ncg */

#endif /* !GRAPH_OPS_NETSRC_H */
