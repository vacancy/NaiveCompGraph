/*
 * graph_op.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#ifndef GRAPH_OP_H
#define GRAPH_OP_H

#include "core/op.h"
#include "core/tensor.h"
#include <cstdint>
#include <string>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <memory>
#include <type_traits>

namespace ncg {

// From graph/graph.h
class Graph;
class GraphForwardContext;

// From this file: graph/op.h
class GraphTensor;
class GraphOp;

class GraphTensor {
public:
    GraphTensor() : m_owner_op(), m_owner_op_index(0), m_desc() {}
    GraphTensor(GraphOp *owner_op, ssize_t index, const TensorDesc &desc) :
        m_owner_op(owner_op), m_owner_op_index(index), m_desc(desc) {
    }
    virtual ~GraphTensor(void) = default;

    template <typename OpType=GraphOp>
    OpType *owner_op(void) {
        return dynamic_cast<OpType *>(m_owner_op);
    }
    template <typename OpType=GraphOp>
    const OpType *owner_op(void) const {
        return dynamic_cast<OpType *>(m_owner_op);
    }

    ssize_t owner_op_index(void) const {
        return m_owner_op_index;
    }
    TensorDesc &desc(void) {
        return m_desc;
    }
    const TensorDesc &desc(void) const {
        return m_desc;
    }

    friend std::ostream & operator << (std::ostream &, const GraphTensor &);

protected:
    GraphOp *m_owner_op;
    ssize_t m_owner_op_index;
    TensorDesc m_desc;
};

typedef std::shared_ptr<GraphTensor> GTensorPtr;
typedef std::vector<GTensorPtr> GTensorVec;

class GraphOp {
public:
    GraphOp() : m_desc(), m_inputs(), m_outputs(), m_initialized(false), m_name(), m_name_initialized(false) {}
    virtual ~GraphOp(void) = default;

    virtual const char *op_name(void) const = 0;
    std::string name(void) const {
        if (m_name_initialized) {
            return m_name;
        }
        return auto_name();
    }
    std::string auto_name(void) const {
        std::ostringstream ss;
        ss << op_name() << "@" << this;
        return ss.str();
    }
    void set_name(const std::string &name) {
        m_name_initialized = true;
        m_name = name;
    }

    template <typename DescT>
    DescT &desc() {
        return *(dynamic_cast<DescT *>(m_desc.get()));
    }
    template <typename DescT>
    const DescT &desc() const {
        return *(dynamic_cast<DescT *>(m_desc.get()));
    }
    const GTensorVec &inputs(void) const {
        return m_inputs;
    }
    const GTensorVec &outputs(void) const {
        return m_outputs;
    }

    const GTensorVec & operator () (Graph &graph, OpDescPtr desc, const GTensorVec &inputs);
    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) = 0;
    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) = 0;
    virtual void forward(GraphForwardContext &ctx) const = 0;

    GTensorPtr make_tensor(ssize_t index, const TensorDesc &desc) {
        return GTensorPtr(new GraphTensor(this, index, desc));
    }

    friend std::ostream & operator << (std::ostream &, const GraphOp &);

protected:
    std::string m_name;
    bool m_name_initialized;

    OpDescPtr m_desc;
    GTensorVec m_inputs;
    GTensorVec m_outputs;
    bool m_initialized;
};

#define NCG_DEF_GOPNAME(op_name_) virtual const char *op_name(void) const { return #op_name_; }

template <typename OpClass>
class GraphOpWrapper : public GraphOp {
public:
    virtual void forward(GraphForwardContext &ctx) const {
        TensorVec inputs;
        for (const auto &gtensor : m_inputs) {
            inputs.push_back(ctx.tensor(gtensor));
        }
        auto op = new OpClass();
        op->set_desc(m_desc);
        TensorVec outputs = op->execute(ctx, inputs);
        ncg_assert(m_outputs.size() == outputs.size());
        for (ssize_t i = 0; i < m_outputs.size(); ++i) {
            ctx.set_tensor(m_outputs[i], outputs[i]);
        }
        delete op;
    }
};

class GraphSingleOutputOp {
};

typedef std::shared_ptr<GraphOp> GOpPtr;

} /* !namespace ncg */

#endif /* !GRAPH_OP_H */
