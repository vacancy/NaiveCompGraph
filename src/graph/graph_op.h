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

class Graph;
class GraphOp;
class GraphTensor;
class GraphForwardContext;

class GraphTensor {
public:
    GraphTensor() : m_owner_op(), m_owner_op_index(0), m_desc() {}
    GraphTensor(GraphOp *owner_op, ssize_t index, const TensorDesc &desc) :
        m_owner_op(owner_op), m_owner_op_index(index), m_desc(desc) {
    }
    virtual ~GraphTensor(void) = default;

    const GraphOp *owner_op(void) const {
        return m_owner_op;
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

class GraphForwardContext : public OpContext {
public:
    GraphForwardContext(Graph &graph) : m_graph(graph), m_feed_dict(), m_storage() {}

    void feed(const std::string &name, TensorPtr tensor) {
        m_feed_dict.emplace(name, tensor);
    }

    TensorPtr feed_dict(const std::string &name) {
        auto it = m_feed_dict.find(name);
        if (it == m_feed_dict.end()) {
            return nullptr;
        }
        return it->second;
    }

    TensorPtr tensor(const GTensorPtr &);
    void set_tensor(const GTensorPtr &gtensor, const TensorPtr &tensor) {
        m_storage.emplace(reinterpret_cast<std::uintptr_t>(gtensor.get()), tensor);
    }

    std::ostringstream &error(const GraphOp *);
protected:
    Graph &m_graph;
    std::unordered_map<std::uintptr_t, TensorPtr> m_storage;
    std::unordered_map<std::string, TensorPtr> m_feed_dict;
};

class GraphOp {
public:
    GraphOp() : m_desc(), m_inputs(), m_outputs(), m_initialized(false), m_name(), m_name_initialized(false) {}
    virtual ~GraphOp(void) = default;

    virtual const char *op_name(void) const = 0;
    std::string name(void) const {
        if (m_name_initialized) {
            return m_name;
        }
        std::ostringstream ss;
        ss << op_name() << "@" << this;
        return ss.str();
    }
    void set_name(const std::string &name) {
        m_name_initialized = true;
        m_name = name;
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

class Graph {
public:
    Graph() : m_ops(), m_is_error(false), m_error() {}
    virtual ~Graph() = default;

    bool ok() const {
        return !m_is_error;
    }

    bool is_error() const {
        return m_is_error;
    }
    std::string error_str() const {
        return m_error.str();
    }
    std::ostringstream &error(const GraphOp *op) {
        m_is_error = true;
        m_error << op->name() << ": ";
        return m_error;
    }

    template <typename OpClass, typename... Tensors>
    GOpPtr make_op(OpDescPtr desc, Tensors &&... args) {
        auto op = new OpClass();
        (*op)(*this, desc, {std::forward<Tensors>(args)...});
        auto op_ptr = GOpPtr(op);
        m_ops.push_back(op_ptr);
        return op_ptr;
    }

    template <typename OpClass, typename... Tensors>
    GOpPtr make_op(const std::string &name, OpDescPtr desc, Tensors &&... args) {
        auto op = new OpClass();
        op->set_name(name);
        (*op)(*this, desc, {std::forward<Tensors>(args)...});
        auto op_ptr = GOpPtr(op);
        m_ops.push_back(op_ptr);
        return op_ptr;
    }

    template <typename OpClass, typename... Tensors>
    typename std::enable_if<std::is_base_of<GraphSingleOutputOp, OpClass>::value, GTensorPtr>::type
    op(OpDescPtr desc, Tensors &&... args) {
        auto op = make_op<OpClass>(desc, std::forward<Tensors>(args)...);
        return op->outputs()[0];
    }

    template <typename OpClass, typename... Tensors>
    typename std::enable_if<std::is_base_of<GraphSingleOutputOp, OpClass>::value, GTensorPtr>::type
    op(const std::string &name, OpDescPtr desc, Tensors &&... args) {
        auto op = make_op<OpClass>(name, desc, std::forward<Tensors>(args)...);
        return op->outputs()[0];
    }

    template <typename OpClass, typename... Tensors>
    typename std::enable_if<!std::is_base_of<GraphSingleOutputOp, OpClass>::value, GTensorVec>::type
    op(OpDescPtr desc, Tensors &&... args) {
        auto op = make_op<OpClass>(desc, std::forward<Tensors>(args)...);
        return op->outputs();
    }

    template <typename OpClass, typename... Tensors>
    typename std::enable_if<!std::is_base_of<GraphSingleOutputOp, OpClass>::value, GTensorVec>::type
    op(const std::string &name, OpDescPtr desc, Tensors &&... args) {
        auto op = make_op<OpClass>(name, desc, std::forward<Tensors>(args)...);
        return op->outputs();
    }

protected:
    std::vector<GOpPtr> m_ops;
    bool m_is_error;
    std::ostringstream m_error;
};

} /* !namespace ncg */

#endif /* !GRAPH_OP_H */
