/*
 * shape.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include "graph/ops/shape.h"
#include "graph/ops/reduction.h"

namespace ncg {


void GOpReshape::backward(Graph &graph, GTensorPtr loss) {
    if (m_inputs.size() == 2) {
        m_inputs[1]->set_grad(graph, loss, nullptr);
    }

    auto output_grad = m_outputs[0]->grad(loss);
    if (output_grad == nullptr) {
        m_inputs[0]->set_grad(graph, loss, nullptr);
        return;
    }

    m_inputs[0]->set_grad(graph, loss, graph.op<GOpReshape>(
        OpDescPtr(new OpReshapeDesc(m_inputs[0]->desc().shape_vec())),
        output_grad,
        graph.op<GOpShapeOf>(nullptr, m_inputs[0])
    ));
}

void GOpPermute::backward(Graph &graph, GTensorPtr loss) {
    auto output_grad = m_outputs[0]->grad(loss);
    if (output_grad == nullptr) {
        m_inputs[0]->set_grad(graph, loss, nullptr);
        return;
    }

    const auto &axes = this->template desc<OpPermuteDesc>().axes;
    auto inverse_axes = ShapeVec(axes.size());
    for (ssize_t i = 0; i < m_inputs[0]->desc().dim(); ++i){
        inverse_axes[axes[i]] = i;
    }
    m_inputs[0]->set_grad(graph, loss, graph.op<GOpPermute>(
        OpDescPtr(new OpPermuteDesc(inverse_axes)),
        output_grad
    ));
}

void GOpExpand::backward(Graph &graph, GTensorPtr loss) {
    if (m_inputs.size() == 2) {
        m_inputs[1]->set_grad(graph, loss, nullptr);
    }

    auto output_grad = m_outputs[0]->grad(loss);
    if (output_grad == nullptr) {
        m_inputs[0]->set_grad(graph, loss, nullptr);
        return;
    }

    auto grad = output_grad;
    for (ssize_t i = 0; i < m_outputs[0]->desc().dim(); ++i) {
        if (m_inputs[0]->desc().shape(i) != m_outputs[0]->desc().shape(i)) {
            grad = graph.op<GOpReduceSum>(
                OpDescPtr(new OpReduceDesc(i, true)),
                grad
            );
        }
    }

    m_inputs[0]->set_grad(graph, loss, grad);
}

void GOpSqueeze::backward(Graph &graph, GTensorPtr loss) {
    auto output_grad = m_outputs[0]->grad(loss);
    if (output_grad == nullptr) {
        m_inputs[0]->set_grad(graph, loss, nullptr);
        return;
    }

    const auto &desc = this->template desc<OpSqueezeDesc>();

    m_inputs[0]->set_grad(graph, loss, graph.op<GOpUnsqueeze>(
        OpDescPtr(new OpUnsqueezeDesc(desc.axis)),
        output_grad
    ));
}

void GOpUnsqueeze::backward(Graph &graph, GTensorPtr loss) {
    auto output_grad = m_outputs[0]->grad(loss);
    if (output_grad == nullptr) {
        m_inputs[0]->set_grad(graph, loss, nullptr);
        return;
    }

    const auto &desc = this->template desc<OpUnsqueezeDesc>();

    m_inputs[0]->set_grad(graph, loss, graph.op<GOpSqueeze>(
        OpDescPtr(new OpSqueezeDesc(desc.axis)),
        output_grad
    ));
}

} /* !namespace ncg */
