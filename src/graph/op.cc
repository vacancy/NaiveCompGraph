/*
 * graph_op.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include "graph_op.h"
#include "graph_forward_impl.h"
#include <memory>

namespace ncg {

std::ostream & operator << (std::ostream &out, const GraphTensor &tensor) {
    out << "GTensor(op=" << tensor.m_owner_op->name() << ", op_type=" << tensor.m_owner_op->op_name() << ", index=" << tensor.m_owner_op_index << ")";
    return out;
}

std::ostream & operator << (std::ostream &out, const GraphOp &op) {
    out << op.name() << "(\n\t";
    for (ssize_t i = 0; i < op.m_inputs.size(); ++i) {
        out << (i == 0 ? "" : ", \n\t") << *(op.m_inputs[i]);
    }
    out  << "\n)";
    return out;
}

TensorPtr GraphForwardContext::tensor(const GTensorPtr &gtensor) {
    auto it = m_storage.find(reinterpret_cast<std::uintptr_t>(gtensor.get()));
    ncg_assert(it != m_storage.end());
    return it->second;
}

std::ostringstream &GraphForwardContext::error(const GraphOp *op) {
    m_is_error = true;
    // m_error << op->name() << ": ";
    return m_error;
}

const GTensorVec &GraphOp::operator () (Graph &graph, OpDescPtr desc, const GTensorVec &inputs) {
    ncg_assert_msg(!m_initialized, std::string("Op ") + name() + " initialized twice.");
    m_initialized = true;

    m_desc = desc;

    check_inputs(graph, inputs);
    m_inputs = inputs;
    if (graph.ok()) {
        m_outputs = init_outputs(graph, inputs);
    }
    return m_outputs;
}

TensorVec GraphForwardContext::eval(const GTensorVec &targets) {
    TensorVec outputs;
    auto sorter = std::make_unique<GraphTopoSorter>(*this);
    sorter->sort(targets);

    for (const GraphOp *op: sorter->sorted()) {
        op->forward(*this);
        if (!ok()) {
            return outputs;
        }
    }

    for (const auto &t: targets) {
        outputs.emplace_back(tensor(t));
    }
    return outputs;
}

} /* !namespace ncg */
