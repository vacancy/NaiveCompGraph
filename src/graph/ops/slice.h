/*
 * slice.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#pragma once

#include "core/tensor_impl.h"
#include "core/ops/slice.h"
#include "graph/op.h"

namespace ncg {

class GOpConcat : public GraphOpWrapper<OpConcat>, public GraphSingleOutputOp {
public:
    NCG_GOP_DEF_NAME(GopConcat);

    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        NCG_OP_CHECK_NONEMPTY_INPUTS(graph, inputs);
        NCG_OP_CHECK_COMPATIBLE_DTYPE(graph, inputs);
        NCG_OP_CHECK_COMPATIBLE_DIM(graph, inputs);
    }

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        auto axis = this->template desc<OpConcatDesc>().axis;
        if (axis < 0) axis += inputs[0]->desc().dim();

        auto shape = inputs[0]->desc().shape_vec();
        for (ssize_t i = 1; i < inputs.size(); ++i) {
            shape[axis] += inputs[i]->desc().shape(axis);
        }

        return {make_tensor(0, TensorDesc(inputs[0]->desc().dtype(), shape))};
    }

    virtual void backward(Graph &graph, GTensorPtr loss);
};

class GOpSplit: public GraphOp {
public:
    NCG_GOP_DEF_NAME(GopSplit);

    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        NCG_OP_CHECK_NR_INPUTS2(graph, inputs, 1, 2);

        if (inputs.size() == 2) {
            NCG_OP_CHECK_INPUT_DTYPE_INT(graph, inputs, 1);
            NCG_OP_CHECK_INPUT_DIM(graph, inputs, 1, 1);
        }
    }

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        const auto &input = inputs[0];
        auto axis = this->template desc<OpSplitDesc>().axis;
        if (axis < 0) axis += inputs[0]->desc().dim();
        const auto &splits = this->template desc<OpSplitDesc>().splits;

        auto outputs = GTensorVec(splits.size());
        ssize_t index = 0;
        for (ssize_t i = 0; i < splits.size(); ++i) {
            auto shape = input->desc().shape_vec();
            shape[axis] = splits[i];
            TensorDesc desc(input->desc().dtype(), shape);
            desc.shape(axis) = splits[i];
            outputs[i] = make_tensor(i, desc);
        }
        return outputs;
    }

    virtual void forward(GraphForwardContext &ctx) const {
        auto input = ctx.tensor(m_inputs[0]);
        const auto &desc = this->template desc<OpSplitDesc>();
        auto axis = desc.axis;
        if (axis < 0) axis += input->desc().dim();

        if (m_inputs.size() == 1) {
            auto op = std::make_unique<OpSplit>();
            op->set_desc(m_desc);
            auto output = op->execute(ctx, {input});
            ctx.set_tensor(m_outputs[0], output[0]);
        } else {
            auto shape_vector = tocc_vector<ssize_t>(ctx.tensor(m_inputs[1]));
            auto tmp_desc = OpDescPtr(new OpSplitDesc(axis, ShapeVec(shape_vector.begin(), shape_vector.end())));
            auto op = std::make_unique<OpSplit>();
            op->set_desc(tmp_desc);
            auto output = op->execute(ctx, {input});
            ctx.set_tensor(m_outputs[0], output[0]);
        }
    }

    virtual void backward(Graph &graph, GTensorPtr loss);
};

class GOpNarrow : public GraphOp, public GraphSingleOutputOp {
public:
    NCG_GOP_DEF_NAME(GOpNarrow);

    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        NCG_OP_CHECK_NR_INPUTS2(graph, inputs, 1, 3);

        if (inputs.size() == 3) {
            NCG_OP_CHECK_INPUT_DTYPE_INT(graph, inputs, 1);
            NCG_OP_CHECK_INPUT_SCALAR(graph, inputs, 1);
            NCG_OP_CHECK_INPUT_DTYPE_INT(graph, inputs, 2);
            NCG_OP_CHECK_INPUT_SCALAR(graph, inputs, 2);
        }
    }

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        const auto &input = inputs[0];
        const auto &desc = this->template desc<OpNarrowDesc>();
        auto axis = desc.axis;
        if (axis < 0) axis += input->desc().dim();
        auto shape = input->desc().shape_vec();
        shape[axis] = desc.length;
        return {make_tensor(0, TensorDesc(input->desc().dtype(), shape))};
    }

    virtual void forward(GraphForwardContext &ctx) const {
        auto input = ctx.tensor(m_inputs[0]);
        const auto &desc = this->template desc<OpNarrowDesc>();
        auto axis = desc.axis;
        if (axis < 0) axis += input->desc().dim();

        if (m_inputs.size() == 1) {
            auto op = std::make_unique<OpNarrow>();
            op->set_desc(m_desc);
            auto output = op->execute(ctx, {input});
            ctx.set_tensor(m_outputs[0], output[0]);
        } else {
            auto start = tocc_scalar<ssize_t>(ctx.tensor(m_inputs[1]));
            auto length = tocc_scalar<ssize_t>(ctx.tensor(m_inputs[2]));
            auto tmp_desc = OpDescPtr(new OpNarrowDesc(axis, start, length));
            auto op = std::make_unique<OpNarrow>();
            op->set_desc(tmp_desc);
            auto output = op->execute(ctx, {input});
            ctx.set_tensor(m_outputs[0], output[0]);
        }
    }

    virtual void backward(Graph &graph, GTensorPtr loss);
};

class GOpNarrowBackward : public GraphOp, public GraphSingleOutputOp {
public:
    NCG_GOP_DEF_NAME(GOpNarrowBackward);

    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        NCG_OP_CHECK_NR_INPUTS2(graph, inputs, 1, 3);

        if (inputs.size() == 3) {
            NCG_OP_CHECK_INPUT_DTYPE_INT(graph, inputs, 1);
            NCG_OP_CHECK_INPUT_SCALAR(graph, inputs, 1);
            NCG_OP_CHECK_INPUT_DTYPE_INT(graph, inputs, 2);
            NCG_OP_CHECK_INPUT_SCALAR(graph, inputs, 2);
        }
    }

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        const auto &input = inputs[0];
        const auto &desc = this->template desc<OpNarrowBackwardDesc>();
        auto axis = desc.axis;
        if (axis < 0) axis += inputs[0]->desc().dim();
        auto shape = input->desc().shape_vec();
        shape[axis] = desc.input_size;
        return {make_tensor(0, TensorDesc(input->desc().dtype(), shape))};
    }

    virtual void forward(GraphForwardContext &ctx) const {
        auto input = ctx.tensor(m_inputs[0]);
        const auto &desc = this->template desc<OpNarrowBackwardDesc>();
        auto axis = desc.axis;
        if (axis < 0) axis += input->desc().dim();

        if (m_inputs.size() == 1) {
            auto op = std::make_unique<OpNarrowBackward>();
            op->set_desc(m_desc);
            auto output = op->execute(ctx, {input});
            ctx.set_tensor(m_outputs[0], output[0]);
        } else {
            auto start = tocc_scalar<ssize_t>(ctx.tensor(m_inputs[1]));
            auto input_size = tocc_scalar<ssize_t>(ctx.tensor(m_inputs[2]));
            auto tmp_desc = OpDescPtr(new OpNarrowBackwardDesc(axis, start, input_size));
            auto op = std::make_unique<OpNarrowBackward>();
            op->set_desc(tmp_desc);
            auto output = op->execute(ctx, {input});
            ctx.set_tensor(m_outputs[0], output[0]);
        }
    }

    virtual void backward(Graph &graph, GTensorPtr loss);
};

class GOpIndexSelect : public GraphOpWrapper<OpIndexSelect>, public GraphSingleOutputOp {
public:
    NCG_GOP_DEF_NAME(GopIndexSelect);

    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        NCG_OP_CHECK_NR_INPUTS(graph, inputs, 2);
        NCG_OP_CHECK_INPUT_DTYPE_INT(graph, inputs, 1);
        NCG_OP_CHECK_INPUT_DIM(graph, inputs, 1, 1);
    }

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        auto axis = this->template desc<OpIndexSelectDesc>().axis;
        if (axis < 0) axis += inputs[0]->desc().dim();

        auto shape = inputs[0]->desc().shape_vec();
        shape[axis] = inputs[1]->desc().shape(0);
        return {make_tensor(0, TensorDesc(inputs[0]->desc().dtype(), shape))};
    }

    virtual void backward(Graph &graph, GTensorPtr loss);
};

class GOpIndexSelectBackward : public GraphOp, public GraphSingleOutputOp {
public:
    NCG_GOP_DEF_NAME(GopIndexSelectBackward);

    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        NCG_OP_CHECK_NR_INPUTS2(graph, inputs, 2, 3);
        NCG_OP_CHECK_INPUT_DTYPE_INT(graph, inputs, 1);
        NCG_OP_CHECK_INPUT_DIM(graph, inputs, 1, 1);

        if (inputs.size() == 3) {
            NCG_OP_CHECK_INPUT_DTYPE_INT(graph, inputs, 2);
            NCG_OP_CHECK_INPUT_SCALAR(graph, inputs, 2);
        }
    }

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        auto axis = this->template desc<OpIndexSelectBackwardDesc>().axis;
        if (axis < 0) axis += inputs[0]->desc().dim();
        auto input_size = this->template desc<OpIndexSelectBackwardDesc>().input_size;

        auto shape = inputs[0]->desc().shape_vec();
        shape[axis] = input_size;
        return {make_tensor(0, TensorDesc(inputs[0]->desc().dtype(), shape))};
    }

    virtual void forward(GraphForwardContext &ctx) const {
        auto input = ctx.tensor(m_inputs[0]);
        auto indices = ctx.tensor(m_inputs[1]);
        const auto &desc = this->template desc<OpIndexSelectBackwardDesc>();
        auto axis = desc.axis;
        if (axis < 0) axis += input->desc().dim();

        if (m_inputs.size() == 2) {
            auto op = std::make_unique<OpIndexSelectBackward>();
            op->set_desc(m_desc);
            auto output = op->execute(ctx, {input, indices});
            ctx.set_tensor(m_outputs[0], output[0]);
        } else {
            auto input_size = tocc_scalar<ssize_t>(ctx.tensor(m_inputs[2]));
            auto tmp_desc = OpDescPtr(new OpIndexSelectBackwardDesc(axis, input_size));
            auto op = std::make_unique<OpIndexSelectBackward>();
            op->set_desc(tmp_desc);
            auto output = op->execute(ctx, {input, indices});
            ctx.set_tensor(m_outputs[0], output[0]);
        }
    }

    virtual void backward(Graph &graph, GTensorPtr loss);
};

class GOpGather : public GraphOpWrapper<OpGather>, public GraphSingleOutputOp {
public:
    NCG_GOP_DEF_NAME(GopGather);

    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        NCG_OP_CHECK_NR_INPUTS(graph, inputs, 2);
        NCG_OP_CHECK_COMPATIBLE_DIM(graph, inputs);
        NCG_OP_CHECK_INPUT_DTYPE_INT(graph, inputs, 1);
    }

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        auto axis = this->template desc<OpGatherDesc>().axis;
        if (axis < 0) axis += inputs[0]->desc().dim();

        auto shape = inputs[0]->desc().shape_vec();
        shape[axis] = inputs[1]->desc().shape(axis);
        return {make_tensor(0, TensorDesc(inputs[0]->desc().dtype(), shape))};
    }

    virtual void backward(Graph &graph, GTensorPtr loss);
};

class GOpGatherBackward : public GraphOp, public GraphSingleOutputOp {
public:
    NCG_GOP_DEF_NAME(GopGatherBackward);

    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        NCG_OP_CHECK_NR_INPUTS2(graph, inputs, 2, 3);
        NCG_OP_CHECK_INPUT_DTYPE_INT(graph, inputs, 1);

        if (inputs.size() == 3) {
            NCG_OP_CHECK_INPUT_DTYPE_INT(graph, inputs, 2);
            NCG_OP_CHECK_INPUT_SCALAR(graph, inputs, 2);
        }
    }

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        auto axis = this->template desc<OpGatherBackwardDesc>().axis;
        if (axis < 0) axis += inputs[0]->desc().dim();
        auto input_size = this->template desc<OpGatherBackwardDesc>().input_size;

        auto shape = inputs[0]->desc().shape_vec();
        shape[axis] = input_size;
        return {make_tensor(0, TensorDesc(inputs[0]->desc().dtype(), shape))};
    }

    virtual void forward(GraphForwardContext &ctx) const {
        auto input = ctx.tensor(m_inputs[0]);
        auto indices = ctx.tensor(m_inputs[1]);
        const auto &desc = this->template desc<OpGatherBackwardDesc>();
        auto axis = desc.axis;
        if (axis < 0) axis += input->desc().dim();

        if (m_inputs.size() == 2) {
            auto op = std::make_unique<OpGatherBackward>();
            op->set_desc(m_desc);
            auto output = op->execute(ctx, {input, indices});
            ctx.set_tensor(m_outputs[0], output[0]);
        } else {
            auto input_size = tocc_scalar<ssize_t>(ctx.tensor(m_inputs[2]));
            auto tmp_desc = OpDescPtr(new OpGatherBackwardDesc(axis, input_size));
            auto op = std::make_unique<OpGatherBackward>();
            op->set_desc(tmp_desc);
            auto output = op->execute(ctx, {input, indices});
            ctx.set_tensor(m_outputs[0], output[0]);
        }
    }

    virtual void backward(Graph &graph, GTensorPtr loss);
};

} /* !namespace ncg */

