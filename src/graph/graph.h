/*
 * graph_forward_impl.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#ifndef GRAPH_GRAPH_H
#define GRAPH_GRAPH_H

#include "graph/op.h"
#include <cstdint>
#include <unordered_set>

namespace ncg {

class GraphTopoSorter {
public:
    GraphTopoSorter(GraphForwardContext &ctx) : m_ctx(ctx) {}

    void sort(const GTensorVec &target) {
        m_sorted.clear();
        m_visited.clear();

        for (const auto &t : target) {
            mark_(t);
        }
    }

    const std::vector<const GraphOp *> &sorted() const {
        return m_sorted;
    }

protected:
    GraphForwardContext &m_ctx;
    std::vector<const GraphOp *> m_sorted;
    std::unordered_set<std::uintptr_t> m_visited;

private:
    void mark_(const GTensorPtr &t) {
        auto op = t->owner_op();
        std::uintptr_t opi = reinterpret_cast<std::uintptr_t>(op);

        if (m_visited.find(opi) != m_visited.end()) {
            return ;
        }
        for (const auto &input : op->inputs()) {
            mark_(input);
        }
        m_sorted.emplace_back(op);
        m_visited.emplace(opi);
    }
};

class GraphForwardContext : public OpContext {
public:
    GraphForwardContext(Graph &graph) : m_graph(graph), m_feed_dict(), m_storage() {}

    std::vector<TensorPtr> eval(const GTensorVec &);

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
        // m_error << op->name() << ": ";
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

#endif /* !GRAPH_GRAPH_H */
