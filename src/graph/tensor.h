/*
 * tensor.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#pragma once

#include "core/tensor.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>
#include <unordered_map>

namespace ncg {

// From graph/op.h
class GraphOp;

// From graph/graph.h
class Graph;


// Forward define the GTensor type.
class GraphTensor;
typedef std::shared_ptr<GraphTensor> GTensorPtr;
typedef std::vector<GTensorPtr> GTensorVec;

class GraphTensor {
public:
    GraphTensor();
    GraphTensor(GraphOp *owner_op, ssize_t index, const TensorDesc &desc);
    virtual ~GraphTensor() = default;

    template <typename OpType=GraphOp>
    OpType *owner_op() {
        return dynamic_cast<OpType *>(m_owner_op);
    }
    template <typename OpType=GraphOp>
    const OpType *owner_op() const {
        return dynamic_cast<OpType *>(m_owner_op);
    }

    ssize_t owner_op_index() const;
    TensorDesc &desc();
    const TensorDesc &desc() const;

    GTensorPtr grad(GTensorPtr loss) const;
    void set_grad(Graph &graph, GTensorPtr loss, GTensorPtr grad);

    friend std::ostream & operator << (std::ostream &, const GraphTensor &);

protected:
    GraphOp *m_owner_op;
    ssize_t m_owner_op_index;
    TensorDesc m_desc;

    std::unordered_map<std::uintptr_t, GTensorPtr> m_grads;
};

} /* !namespace ncg */

