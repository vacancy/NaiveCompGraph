/*
 * netsrc.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#pragma once

#include "graph/op.h"
#include <vector>

namespace ncg {

class GraphNetSrcOp : public GraphOp, public GraphSingleOutputOp {
public:
    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        NCG_OP_CHECK_EMPTY_INPUTS(graph, inputs);
    }

    NCG_GOP_DEF_NO_GRAD_INLINE;
};

class GOpPlaceholderDesc : public OpDesc {
public:
    GOpPlaceholderDesc() : desc() {}
    GOpPlaceholderDesc(DTypeName dtype, const ShapeVec &shape) : desc(dtype, shape) {}
    virtual ~GOpPlaceholderDesc() = default;

    TensorDesc desc;
};

class GOpPlaceholder : public GraphNetSrcOp {
public:
    NCG_GOP_DEF_NAME(GOpPlaceholder);

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
    NCG_GOP_DEF_NAME(GOpConstant);

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
    NCG_GOP_DEF_NAME(GOpVariable);

    void set_value(Session &session, const TensorPtr &tensor) {
        session.set_shared_tensor(m_outputs[0], tensor);
    }
    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        return {make_tensor(0, this->template desc<GOpVariableDesc>().tensor->desc())};
    }
    virtual void forward(GraphForwardContext &ctx) const {
        if (!ctx.session().is_shared_tensor_initialized(m_outputs[0])) {
            ctx.session().set_shared_tensor(m_outputs[0], this->template desc<GOpVariableDesc>().tensor);
        }
        ctx.set_tensor(m_outputs[0], ctx.session().shared_tensor(m_outputs[0]));
    }
};

class GraphNetSrcOpDynamicShape : public GraphOp, public GraphSingleOutputOp {
public:
    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        NCG_OP_CHECK_NR_INPUTS2(graph, inputs, 0, 1);
        if (inputs.size() == 1) {
            NCG_OP_CHECK_INPUT_VECTOR(graph, inputs, 0);
        }
    }

    NCG_GOP_DEF_NO_GRAD_INLINE;
};

class OpZerosDesc : public OpDesc {
public:
    OpZerosDesc() : desc() {}
    OpZerosDesc(DTypeName dtype, const ShapeVec &shape) : desc(dtype, shape) {}
    virtual ~OpZerosDesc() = default;

    TensorDesc desc;
};

class GOpZeros: public GraphNetSrcOpDynamicShape {
public:
    NCG_GOP_DEF_NAME(GOpZeros);

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        return {make_tensor(0, this->template desc<OpZerosDesc>().desc)};
    }
    virtual void forward(GraphForwardContext &ctx) const {
        auto &desc = this->template desc<OpZerosDesc>().desc;

        ShapeVec shape;
        if (m_inputs.size() == 0) {
            shape = desc.shape_vec();
        } else {
            auto tmp_shape = tocc_vector<ssize_t>(ctx.tensor(m_inputs[0]));
            shape = ShapeVec(tmp_shape.begin(), tmp_shape.end());
        }
        auto tensor = zeros(desc.dtype(), shape);
        ctx.set_tensor(m_outputs[0], tensor);
    }
};

class OpOnesDesc : public OpDesc {
public:
    OpOnesDesc() : desc() {}
    OpOnesDesc(DTypeName dtype, const ShapeVec &shape) : desc(dtype, shape) {}
    virtual ~OpOnesDesc() = default;

    TensorDesc desc;
};

class GOpOnes: public GraphNetSrcOpDynamicShape {
public:
    NCG_GOP_DEF_NAME(GOpZeros);

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        return {make_tensor(0, this->template desc<OpOnesDesc>().desc)};
    }
    virtual void forward(GraphForwardContext &ctx) const {
        auto &desc = this->template desc<OpOnesDesc>().desc;
        ShapeVec shape;
        if (m_inputs.size() == 0) {
            shape = desc.shape_vec();
        } else {
            auto tmp_shape = tocc_vector<ssize_t>(ctx.tensor(m_inputs[0]));
            shape = ShapeVec(tmp_shape.begin(), tmp_shape.end());
        }
        auto tensor = ones(desc.dtype(), shape);
        ctx.set_tensor(m_outputs[0], tensor);
    }
};

} /* !namespace ncg */

