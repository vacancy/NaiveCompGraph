/*
 * linalg.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include "graph/ops/linalg.h"

namespace ncg {

void GOpMatMul::backward(Graph &graph, GTensorPtr loss) {
    const auto &desc = this->template desc<OpMatMulDesc>();

    auto output_grad = m_outputs[0]->grad(loss);
    if (output_grad == nullptr) {
        m_inputs[0]->set_grad(graph, loss, nullptr);
        m_inputs[1]->set_grad(graph, loss, nullptr);
        return;
    }

    m_inputs[0]->set_grad(graph, loss,
        graph.op<GOpMatMul>(OpDescPtr(new OpMatMulDesc(
            desc.transpose_a, !desc.transpose_b
        )), output_grad, m_inputs[1])
    );
    m_inputs[1]->set_grad(graph, loss,
        graph.op<GOpMatMul>(OpDescPtr(new OpMatMulDesc(
            !desc.transpose_a, desc.transpose_b
        )), m_inputs[0], output_grad)
    );
}

} /* !namespace ncg */
