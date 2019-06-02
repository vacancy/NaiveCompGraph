/*
 * graph_op.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include "graph/tensor.h"
#include "graph/op.h"
#include "graph/graph.h"

#include <sstream>

namespace ncg {

GraphOp::GraphOp() :
    m_desc(), m_inputs(), m_outputs(),
    m_initialized(false),
    m_name(), m_name_initialized(false) {
    // Pass
}

std::string GraphOp::name(void) const {
    if (m_name_initialized) {
        return m_name;
    }
    return auto_name();
}

std::string GraphOp::auto_name(void) const {
    std::ostringstream ss;
    ss << op_name() << "@" << this;
    return ss.str();
}

void GraphOp::set_name(const std::string &name) {
    m_name_initialized = true;
    m_name = name;
}

const GTensorVec &GraphOp::inputs(void) const {
    return m_inputs;
}

const GTensorVec &GraphOp::outputs(void) const {
    return m_outputs;
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

GTensorPtr GraphOp::make_tensor(ssize_t index, const TensorDesc &desc) {
    return GTensorPtr(new GraphTensor(this, index, desc));
}

std::ostream & operator << (std::ostream &out, const GraphOp &op) {
    out << op.name() << "(\n\t";
    for (ssize_t i = 0; i < op.m_inputs.size(); ++i) {
        out << (i == 0 ? "" : ", \n\t") << *(op.m_inputs[i]);
    }
    out  << "\n)";
    return out;
}

} /* !namespace ncg */
