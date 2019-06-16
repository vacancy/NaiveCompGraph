/*
 * reduction.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include "graph/ops/elemwise.h"
#include "graph/ops/reduction.h"
#include "graph/ops/shape.h"
#include "graph/ops/slice.h"

namespace ncg {

void GOpReduceMax::backward(Graph &graph, GTensorPtr loss) {
    auto output_grad = m_outputs[0]->grad(loss);
    if (output_grad == nullptr) {
        m_inputs[0]->set_grad(graph, loss, nullptr);
        return;
    }

    const auto &desc = this->template desc<OpReduceDesc>();

    if (!desc.keepdims) {
        output_grad = graph.op<GOpUnsqueeze>(
            OpDescPtr(new OpUnsqueezeDesc(desc.axis)),
            output_grad
        );
    }

    m_inputs[0]->set_grad(graph, loss, graph.op<GOpGatherBackward>(
        OpDescPtr(new OpGatherBackwardDesc(desc.axis, m_inputs[0]->desc().shape(desc.axis))),
        output_grad, m_outputs[1],
        graph.op<GOpShapeOfIndex>(OpDescPtr(new OpShapeOfIndexDesc(desc.axis)), m_inputs[0])
    ));
}

void GOpReduceMin::backward(Graph &graph, GTensorPtr loss) {
    auto output_grad = m_outputs[0]->grad(loss);
    if (output_grad == nullptr) {
        m_inputs[0]->set_grad(graph, loss, nullptr);
        return;
    }

    const auto &desc = this->template desc<OpReduceDesc>();

    if (!desc.keepdims) {
        output_grad = graph.op<GOpUnsqueeze>(
            OpDescPtr(new OpUnsqueezeDesc(desc.axis)),
            output_grad
        );
    }

    m_inputs[0]->set_grad(graph, loss, graph.op<GOpGatherBackward>(
        OpDescPtr(new OpGatherBackwardDesc(desc.axis, m_inputs[0]->desc().shape(desc.axis))),
        output_grad, m_outputs[1],
        graph.op<GOpShapeOfIndex>(OpDescPtr(new OpShapeOfIndexDesc(desc.axis)), m_inputs[0])
    ));
}

void GOpReduceSum::backward(Graph &graph, GTensorPtr loss) {
    auto output_grad = m_outputs[0]->grad(loss);
    if (output_grad == nullptr) {
        m_inputs[0]->set_grad(graph, loss, nullptr);
        return;
    }

    const auto &desc = this->template desc<OpReduceDesc>();

    if (!desc.keepdims) {
        output_grad = graph.op<GOpUnsqueeze>(
            OpDescPtr(new OpUnsqueezeDesc(desc.axis)),
            output_grad
        );
    }

    m_inputs[0]->set_grad(graph, loss, graph.op<GOpExpand>(
        OpDescPtr(new OpExpandDesc(m_inputs[0]->desc().shape_vec())),
        output_grad,
        graph.op<GOpShapeOf>(nullptr, m_inputs[0])
    ));
}

void GOpReduceMean::backward(Graph &graph, GTensorPtr loss) {
    auto output_grad = m_outputs[0]->grad(loss);
    if (output_grad == nullptr) {
        m_inputs[0]->set_grad(graph, loss, nullptr);
        return;
    }

    const auto &desc = this->template desc<OpReduceDesc>();

    if (!desc.keepdims) {
        output_grad = graph.op<GOpUnsqueeze>(
            OpDescPtr(new OpUnsqueezeDesc(desc.axis)),
            output_grad
        );
    }

    m_inputs[0]->set_grad(graph, loss, graph.op<GOpDiv>(nullptr,
        G::auto_broadcast(graph, {
            graph.op<GOpExpand>(
                OpDescPtr(new OpExpandDesc(m_inputs[0]->desc().shape_vec())),
                output_grad,
                graph.op<GOpShapeOf>(nullptr, m_inputs[0])
            ),
            graph.op<GOpCast>(
                OpDescPtr(new OpCastDesc(m_inputs[0]->desc().dtype())),
                graph.op<GOpShapeOfIndex>(
                    OpDescPtr(new OpShapeOfIndexDesc(desc.axis)),
                    m_inputs[0]
                )
            )
        })
    ));
}

} /* !namespace ncg */
