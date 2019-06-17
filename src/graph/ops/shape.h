/*
 * shape.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#pragma once

#include "core/tensor.h"
#include "core/tensor_impl.h"
#include "core/ops/shape.h"

#include "graph/op.h"

namespace ncg {

class GOpShapeOf : public GraphOp, public GraphSingleOutputOp {
public:
    NCG_GOP_DEF_NAME(GOpShapeOf);

    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        NCG_OP_CHECK_NR_INPUTS(graph, inputs, 1);
    }

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        return {make_tensor(0, TensorDesc(DTypeName::Int64, {static_cast<ssize_t>(inputs[0]->desc().dim())}))};
    }

    virtual void forward(GraphForwardContext &ctx) const {
        auto input = ctx.tensor(m_inputs[0]);
        auto shape_tensor = fromcc(DTypeName::Int64, input->desc().shape_vec());
        ctx.set_tensor(m_outputs[0], shape_tensor);
    }

    NCG_GOP_DEF_NO_GRAD_INLINE;
};

class OpShapeOfIndexDesc : public OpDesc {
public:
    OpShapeOfIndexDesc() : axis(0) {}
    OpShapeOfIndexDesc(ssize_t axis) : axis(axis) {}
    virtual ~OpShapeOfIndexDesc() = default;

    ssize_t axis;
};

class GOpShapeOfIndex : public GraphOp, public GraphSingleOutputOp {
public:
    NCG_GOP_DEF_NAME(GOpShapeOfIndex);

    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        NCG_OP_CHECK_NR_INPUTS(graph, inputs, 1);
        const auto &desc = this->template desc<OpShapeOfIndexDesc>();
        auto axis = desc.axis;
        if (axis < 0) axis += inputs[0]->desc().dim();
        NCG_OP_CHECK_INPUT_DIM_GEQ(graph, inputs, 0, axis);
    }

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        return {make_tensor(0, TensorDesc(DTypeName::Int64, {}))};
    }

    virtual void forward(GraphForwardContext &ctx) const {
        auto input = ctx.tensor(m_inputs[0]);
        const auto &desc = this->template desc<OpShapeOfIndexDesc>();
        auto axis = desc.axis;
        if (axis < 0) axis += input->desc().dim();

        auto shape_tensor = fromcc(DTypeName::Int64, input->desc().shape(axis));
        ctx.set_tensor(m_outputs[0], shape_tensor);
    }

    NCG_GOP_DEF_NO_GRAD_INLINE;
};

class GOpShapeConcat : public GraphOp, public GraphSingleOutputOp {
public:
    NCG_GOP_DEF_NAME(GOpShapeConcat);

    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        NCG_OP_CHECK_NONEMPTY_INPUTS(graph, inputs);
        for (ssize_t i = 0; i < inputs.size(); ++i) {
            NCG_OP_CHECK_INPUT_SCALAR_VECTOR(graph, inputs, i);
            NCG_OP_CHECK_INPUT_DTYPE_INT(graph, inputs, i);
        }
    }

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        ssize_t tot = 0;
        for (const auto &i: inputs) {
            if (i->desc().dim() == 0) ++tot;
            else tot += i->desc().shape(0);
        }
        return {make_tensor(0, TensorDesc(DTypeName::Int64, {tot}))};
    }

    virtual void forward(GraphForwardContext &ctx) const {
        ssize_t tot = 0;
        for (const auto &gi: m_inputs) {
            auto i = ctx.tensor(gi);
            if (i->desc().dim() == 0) ++tot;
            else tot += i->desc().shape(0);
        }

        auto shape_tensor = empty(DTypeName::Int64, {tot});
        auto shape_tensor_dtype = shape_tensor->template as<DTypeName::Int64>();
        tot = 0;
        for (const auto &gi: m_inputs) {
            auto i = ctx.tensor(gi);
            if (i->desc().dim() == 0) {
                auto i_value = tocc_scalar<ssize_t>(i);
                shape_tensor_dtype->mutable_at(tot) = i_value;
                ++tot;
            } else {
                auto i_value = tocc_vector<ssize_t>(i);
                for (ssize_t j = 0; j < i->desc().shape(0); ++j) {
                    shape_tensor_dtype->mutable_at(tot) = i_value[j];
                    ++tot;
                }
            }
        }

        ctx.set_tensor(m_outputs[0], shape_tensor);
    }

    NCG_GOP_DEF_NO_GRAD_INLINE;
};

class GOpReshape : public GraphOp, public GraphSingleOutputOp {
public:
    NCG_GOP_DEF_NAME(GOpReshape);

    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        NCG_OP_CHECK_NR_INPUTS2(graph, inputs, 1, 2);

        if (inputs.size() == 2) {
            NCG_OP_CHECK_INPUT_DTYPE_INT(graph, inputs, 1);
            NCG_OP_CHECK_INPUT_DIM(graph, inputs, 1, 1);
        }
    }

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        const auto &desc = this->template desc<OpReshapeDesc>();
        return {make_tensor(0, TensorDesc(inputs[0]->desc().dtype(), desc.shape))};
    }

    virtual void forward(GraphForwardContext &ctx) const {
        const auto &desc = this->template desc<OpReshapeDesc>();
        auto input = ctx.tensor(m_inputs[0]);

        if (m_inputs.size() == 1) {
            auto op = std::make_unique<OpReshape>();
            op->set_desc(m_desc);
            auto output = op->execute(ctx, {input});
            ctx.set_tensor(m_outputs[0], output[0]);
        } else {
            auto shape_vector = tocc_vector<ssize_t>(ctx.tensor(m_inputs[1]));
            auto tmp_desc = OpDescPtr(new OpReshapeDesc(ShapeVec(shape_vector.begin(), shape_vector.end())));
            auto op = std::make_unique<OpReshape>();
            op->set_desc(tmp_desc);
            auto output = op->execute(ctx, {input});
            ctx.set_tensor(m_outputs[0], output[0]);
        }
    }

    virtual void backward(Graph &graph, GTensorPtr loss);
};

class GOpPermute : public GraphOp, public GraphSingleOutputOp {
public:
    NCG_GOP_DEF_NAME(GOpPermute);

    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        NCG_OP_CHECK_NR_INPUTS(graph, inputs, 1);
    }

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        const auto input = inputs[0];
        const auto &axes = this->template desc<OpPermuteDesc>().axes;

        ShapeVec output_shape = input->desc().shape_vec();
        for (ssize_t i = 0; i < input->desc().dim(); ++i) {
            output_shape[i] = input->desc().shape(axes[i]);
        }
        return {make_tensor(0, TensorDesc(input->desc().dtype(), output_shape))};
    }

    virtual void forward(GraphForwardContext &ctx) const {
        const auto &desc = this->template desc<OpPermuteDesc>();
        auto input = ctx.tensor(m_inputs[0]);
        auto op = std::make_unique<OpPermute>();
        op->set_desc(m_desc);
        auto output = op->execute(ctx, {input});
        ctx.set_tensor(m_outputs[0], output[0]);
    }

    virtual void backward(Graph &graph, GTensorPtr loss);
};

class GOpExpand : public GraphOp, public GraphSingleOutputOp {
public:
    NCG_GOP_DEF_NAME(GOpExpand);

    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        NCG_OP_CHECK_NR_INPUTS2(graph, inputs, 1, 2);

        if (inputs.size() == 2) {
            NCG_OP_CHECK_INPUT_DTYPE_INT(graph, inputs, 1);
            NCG_OP_CHECK_INPUT_DIM(graph, inputs, 1, 1);
        }
    }

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        const auto &desc = this->template desc<OpExpandDesc>();
        return {make_tensor(0, TensorDesc(inputs[0]->desc().dtype(), desc.shape))};
    }

    virtual void forward(GraphForwardContext &ctx) const {
        const auto &desc = this->template desc<OpExpandDesc>();
        auto input = ctx.tensor(m_inputs[0]);

        if (m_inputs.size() == 1) {
            auto op = std::make_unique<OpExpand>();
            op->set_desc(m_desc);
            auto output = op->execute(ctx, {input});
            ctx.set_tensor(m_outputs[0], output[0]);
        } else {
            auto shape_vector = tocc_vector<ssize_t>(ctx.tensor(m_inputs[1]));
            auto tmp_desc = OpDescPtr(new OpExpandDesc(ShapeVec(shape_vector.begin(), shape_vector.end())));
            auto op = std::make_unique<OpExpand>();
            op->set_desc(tmp_desc);
            auto output = op->execute(ctx, {input});
            ctx.set_tensor(m_outputs[0], output[0]);
        }
    }

    virtual void backward(Graph &graph, GTensorPtr loss);
};

class GOpSqueeze : public GraphOpWrapper<OpSqueeze>, public GraphSingleOutputOp {
public:
    NCG_GOP_DEF_NAME(GOpSqueeze);

    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        NCG_OP_CHECK_NR_INPUTS(graph, inputs, 1);

        const auto &desc = this->template desc<OpSqueezeDesc>();
        auto axis = desc.axis;
        if (axis < 0) axis += inputs[0]->desc().dim();
        NCG_OP_CHECK_INPUT_DIM_GEQ(graph, inputs, 0, axis);
    }

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        const auto &desc = this->template desc<OpSqueezeDesc>();
        auto axis = desc.axis;
        if (axis < 0) axis += inputs[0]->desc().dim();

        auto shape = inputs[0]->desc().shape_vec();
        shape.erase(shape.begin() + axis);

        return {make_tensor(0, TensorDesc(inputs[0]->desc().dtype(), shape))};
    }

    virtual void backward(Graph &graph, GTensorPtr loss);
};

class GOpUnsqueeze : public GraphOpWrapper<OpUnsqueeze>, public GraphSingleOutputOp {
public:
    NCG_GOP_DEF_NAME(GOpUnsqueeze);

    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        NCG_OP_CHECK_NR_INPUTS(graph, inputs, 1);

        const auto &desc = this->template desc<OpUnsqueezeDesc>();
        auto axis = desc.axis;
        if (axis < 0) axis += inputs[0]->desc().dim();
        NCG_OP_CHECK_INPUT_DIM_GEQ(graph, inputs, 0, axis - 1);
    }

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        const auto &desc = this->template desc<OpUnsqueezeDesc>();
        auto axis = desc.axis;
        if (axis < 0) axis += inputs[0]->desc().dim();

        auto shape = inputs[0]->desc().shape_vec();
        shape.insert(shape.begin() + axis, 1);

        return {make_tensor(0, TensorDesc(inputs[0]->desc().dtype(), shape))};
    }

    virtual void backward(Graph &graph, GTensorPtr loss);
};

// auto broadcast

} /* !namespace ncg */

