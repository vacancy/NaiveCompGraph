/*
 * slice.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include "graph/ops/slice.h"
#include "graph/ops/netsrc.h"
#include "graph/ops/shape.h"

namespace ncg {

void GOpConcat::backward(Graph &graph, GTensorPtr loss) {
    auto output_grad = m_outputs[0]->grad(loss);
    if (output_grad == nullptr) {
        for (auto &i : m_inputs) {
            i->set_grad(graph, loss, nullptr);
        }
        return;
    }

    const auto &desc = this->template desc<OpConcatDesc>();
    auto sta_splits = ShapeVec();
    auto dyn_splits = GTensorVec();
    for (auto &i : m_inputs) {
        sta_splits.emplace_back(i->desc().shape(desc.axis));
        dyn_splits.push_back(graph.op<GOpShapeOfIndex>(OpDescPtr(new OpShapeOfIndexDesc(desc.axis)), i));
    }

    auto input_grads = graph.op<GOpSplit>(
        OpDescPtr(new OpSplitDesc(desc.axis, sta_splits)),
        output_grad, graph.op<GOpShapeConcat>(nullptr, dyn_splits)
    );

    for (ssize_t i = 0; i < m_inputs.size(); ++i) {
        m_inputs[i]->set_grad(graph, loss, input_grads[i]);
    }
}

void GOpSplit::backward(Graph &graph, GTensorPtr loss) {
    if (m_inputs.size() == 2) {
        m_inputs[1]->set_grad(graph, loss, nullptr);
    }

    GTensorVec output_grads;
    bool is_null = true;
    for (auto &o : m_outputs) {
        auto grad = o->grad(loss);
        if (grad != nullptr) {
            is_null = false;
        }
    }

    if (is_null) {
        m_inputs[0]->set_grad(graph, loss, nullptr);
        return;
    }

    for (auto &o : m_outputs) {
        auto grad = o->grad(loss);
        if (grad == nullptr) {
            grad = graph.op<GOpZeros>(
                OpDescPtr(new OpZerosDesc(o->desc().dtype(), o->desc().shape_vec())),
                graph.op<GOpShapeOf>(nullptr, o)
            );
        }
        output_grads.push_back(grad);
    }

    const auto &desc = this->template desc<OpSplitDesc>();
    m_inputs[0]->set_grad(graph, loss, graph.op<GOpConcat>(
        OpDescPtr(new OpConcatDesc(desc.axis)),
        output_grads
    ));
}

void GOpNarrow::backward(Graph &graph, GTensorPtr loss) {
    if (m_inputs.size() == 3) {
        m_inputs[1]->set_grad(graph, loss, nullptr);
        m_inputs[2]->set_grad(graph, loss, nullptr);
    }

    auto output_grad = m_outputs[0]->grad(loss);
    if (output_grad == nullptr) {
        m_inputs[0]->set_grad(graph, loss, nullptr);
        return;
    }

    const auto &desc = this->template desc<OpNarrowDesc>();
    m_inputs[0]->set_grad(graph, loss, graph.op<GOpNarrowBackward>(
        OpDescPtr(new OpNarrowBackwardDesc(desc.axis, desc.start, m_inputs[0]->desc().shape(desc.axis))),
        output_grad,
        as_gtensor<int64_t>(graph, desc.start),
        graph.op<GOpShapeOfIndex>(OpDescPtr(new OpShapeOfIndexDesc(desc.axis)), m_inputs[0])
    ));
}

void GOpNarrowBackward::backward(Graph &graph, GTensorPtr loss) {
    if (m_inputs.size() == 3) {
        m_inputs[1]->set_grad(graph, loss, nullptr);
        m_inputs[2]->set_grad(graph, loss, nullptr);
    }

    auto output_grad = m_outputs[0]->grad(loss);
    if (output_grad == nullptr) {
        m_inputs[0]->set_grad(graph, loss, nullptr);
        return;
    }

    const auto &desc = this->template desc<OpNarrowBackwardDesc>();
    m_inputs[0]->set_grad(graph, loss, graph.op<GOpNarrow>(
        OpDescPtr(new OpNarrowDesc(desc.axis, desc.start, m_inputs[0]->desc().shape(desc.axis))),
        output_grad,
        as_gtensor<int64_t>(graph, desc.start),
        graph.op<GOpShapeOfIndex>(OpDescPtr(new OpShapeOfIndexDesc(desc.axis)), m_inputs[0])
    ));
}

void GOpIndexSelect::backward(Graph &graph, GTensorPtr loss) {
    m_inputs[1]->set_grad(graph, loss, nullptr);

    auto output_grad = m_outputs[0]->grad(loss);
    if (output_grad == nullptr) {
        m_inputs[0]->set_grad(graph, loss, nullptr);
        return;
    }

    const auto &desc = this->template desc<OpIndexSelectDesc>();
    m_inputs[0]->set_grad(graph, loss, graph.op<GOpIndexSelectBackward>(
        OpDescPtr(new OpIndexSelectBackwardDesc(desc.axis, m_inputs[0]->desc().shape(desc.axis))),
        output_grad, m_inputs[1],
        graph.op<GOpShapeOfIndex>(OpDescPtr(new OpShapeOfIndexDesc(desc.axis)), m_inputs[0])
    ));
}

void GOpIndexSelectBackward::backward(Graph &graph, GTensorPtr loss) {
    m_inputs[1]->set_grad(graph, loss, nullptr);

    if (m_inputs.size() == 3) {
        m_inputs[2]->set_grad(graph, loss, nullptr);
    }

    auto output_grad = m_outputs[0]->grad(loss);
    if (output_grad == nullptr) {
        m_inputs[0]->set_grad(graph, loss, nullptr);
        return;
    }

    const auto &desc = this->template desc<OpIndexSelectBackwardDesc>();
    m_inputs[0]->set_grad(graph, loss, graph.op<GOpIndexSelect>(
        OpDescPtr(new OpIndexSelectDesc(desc.axis)),
        output_grad, m_inputs[1]
    ));
}

void GOpGather::backward(Graph &graph, GTensorPtr loss) {
    m_inputs[1]->set_grad(graph, loss, nullptr);

    auto output_grad = m_outputs[0]->grad(loss);
    if (output_grad == nullptr) {
        m_inputs[0]->set_grad(graph, loss, nullptr);
        return;
    }

    const auto &desc = this->template desc<OpGatherDesc>();
    m_inputs[0]->set_grad(graph, loss, graph.op<GOpGatherBackward>(
        OpDescPtr(new OpGatherBackwardDesc(desc.axis, m_inputs[0]->desc().shape(desc.axis))),
        output_grad, m_inputs[1]
    ));
}

void GOpGatherBackward::backward(Graph &graph, GTensorPtr loss) {
    m_inputs[1]->set_grad(graph, loss, nullptr);

    if (m_inputs.size() == 3) {
        m_inputs[2]->set_grad(graph, loss, nullptr);
    }

    auto output_grad = m_outputs[0]->grad(loss);
    if (output_grad == nullptr) {
        m_inputs[0]->set_grad(graph, loss, nullptr);
        return;
    }

    const auto &desc = this->template desc<OpGatherBackwardDesc>();
    m_inputs[0]->set_grad(graph, loss, graph.op<GOpGather>(
        OpDescPtr(new OpGatherDesc(desc.axis)),
        output_grad, m_inputs[1],
        graph.op<GOpShapeOfIndex>(OpDescPtr(new OpShapeOfIndexDesc(desc.axis)), m_inputs[0])
    ));
}

} /* !namespace ncg */

