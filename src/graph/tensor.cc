/*
 * tensor.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include "graph/tensor.h"
#include "graph/op.h"

namespace ncg {

GraphTensor::GraphTensor() : m_owner_op(), m_owner_op_index(0), m_desc() {
    // Pass
}

GraphTensor::GraphTensor(GraphOp *owner_op, ssize_t index, const TensorDesc &desc) :
    m_owner_op(owner_op), m_owner_op_index(index), m_desc(desc) {
    // Pass
}

ssize_t GraphTensor::owner_op_index(void) const {
    return m_owner_op_index;
}

TensorDesc &GraphTensor::desc(void) {
    return m_desc;
}

const TensorDesc &GraphTensor::desc(void) const {
    return m_desc;
}

GTensorPtr grad(GTensorPtr loss) const {
    auto tensor = loss.get();
    std::uintptr_t tpi = reinterpret_cast<std::uintptr_t>(tensor);

    auto it = m_grads.find(tpi);
    if (it == m_grads.end()) {
        return nullptr;
    }
    return it->second;
}

void set_grad(GTensorPtr loss, GTensorPtr grad) {
    auto tensor = loss.get();
    std::uintptr_t tpi = reinterpret_cast<std::uintptr_t>(tensor);
    m_grads.emplace(tpi, grad);
}

std::ostream & operator << (std::ostream &out, const GraphTensor &tensor) {
    out << "GTensor(op=" << tensor.m_owner_op->name() << ", op_type=" << tensor.m_owner_op->op_name() << ", index=" << tensor.m_owner_op_index << ")";
    return out;
}

} /* !namespace ncg */
