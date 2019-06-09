/*
 * graph.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#ifndef GRAPH_GRAPH_H
#define GRAPH_GRAPH_H

#include "graph/tensor.h"
#include "graph/op.h"

#include <cstdint>
#include <string>
#include <sstream>
#include <memory>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <type_traits>

namespace ncg {

class GraphTopoSorter final {
public:
    GraphTopoSorter(Graph &graph);

    void sort(const GTensorVec &target);
    const std::vector<GraphOp *> &sorted() const;

protected:
    Graph &m_graph;
    std::vector<GraphOp *> m_sorted;
    std::unordered_set<std::uintptr_t> m_visited;

private:
    void mark_(const GTensorPtr &t);
};

class Graph {
public:
    Graph();
    virtual ~Graph() = default;

    bool ok() const;
    bool is_error() const;
    std::string error_str() const;
    std::ostringstream &error(const GraphOp *op);

    virtual void backward(GTensorPtr loss);

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

    GOpPtr find_op(const std::string &name);

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

class Session {
public:
    Session(Graph &graph);

    Graph &graph();
    const Graph &graph() const;

    TensorPtr shared_tensor(const GTensorPtr &);
    void set_shared_tensor(const GTensorPtr &gtensor, const TensorPtr &tensor);

protected:
    Graph &m_graph;
    std::unordered_map<std::uintptr_t, TensorPtr> m_shared_tensors;
};

class GraphForwardContext : public OpContext {
public:
    GraphForwardContext(Session &session);

    void feed(const std::string &name, TensorPtr tensor);
    TensorPtr feed_dict(const std::string &name);
    std::vector<TensorPtr> eval(const GTensorVec &);

    TensorPtr tensor(const GTensorPtr &);
    void set_tensor(const GTensorPtr &gtensor, const TensorPtr &tensor);

    std::ostringstream &error(const GraphOp *);

protected:
    Session &m_session;
    std::unordered_map<std::uintptr_t, TensorPtr> m_storage;
    std::unordered_map<std::string, TensorPtr> m_feed_dict;
};

} /* !namespace ncg */

#endif /* !GRAPH_GRAPH_H */
