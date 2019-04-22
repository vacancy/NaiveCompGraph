/*
 * graph_op.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include "graph_op.h"

namespace ncg {

std::ostream & operator << (std::ostream &out, const GraphTensor &tensor) {
    out << "GTensor(op=" << tensor.m_owner_op->name() << ", index=" << tensor.m_owner_op_index << ")";
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
    if (it == m_storage.end()) {
        /* TODO: topo sort. */
        gtensor->owner_op()->forward(*this);
        it = m_storage.find(reinterpret_cast<std::uintptr_t>(gtensor.get()));
        ncg_assert(it != m_storage.end());
        return it->second;
    }
    return it->second;
}

std::ostringstream &GraphForwardContext::error(const GraphOp *op) {
    m_is_error = true;
    m_error << op->name() << ": ";
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


} /* !namespace ncg */
