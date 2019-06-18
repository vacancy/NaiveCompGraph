/*
 * graph.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#pragma once

#include "core/op.h"
#include "graph/tensor.h"

#include <cstdint>
#include <unordered_set>
#include <unordered_map>

namespace ncg {

class GraphOp;
class GraphSingleOutputOp;
typedef std::shared_ptr<GraphOp> GOpPtr;

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

class Graph : public RuntimeContext {
public:
    Graph();
    virtual ~Graph() = default;

    std::ostringstream &error(const GraphOp *op);

    virtual void backward(GTensorPtr loss);

    template <typename OpClass>
    GOpPtr make_op(OpDescPtr desc, const TensorVec &inputs) {
        auto op = new OpClass();
        (*op)(*this, desc, inputs);
        ncg_assert_msg(ok(), error_str());
        auto op_ptr = GOpPtr(op);
        m_ops.push_back(op_ptr);
        return op_ptr;
    }

    template <typename OpClass, typename... Tensors>
    GOpPtr make_op(OpDescPtr desc, Tensors &&... args) {
        auto op = new OpClass();
        (*op)(*this, desc, {std::forward<Tensors>(args)...});
        ncg_assert_msg(ok(), error_str());
        auto op_ptr = GOpPtr(op);
        m_ops.push_back(op_ptr);
        return op_ptr;
    }

    template <typename OpClass, typename... Tensors>
    GOpPtr make_op(const std::string &name, OpDescPtr desc, const TensorVec &inputs) {
        auto op = new OpClass();
        op->set_name(name);
        (*op)(*this, desc, inputs);
        ncg_assert_msg(ok(), error_str());
        auto op_ptr = GOpPtr(op);
        m_ops.push_back(op_ptr);
        return op_ptr;
    }

    template <typename OpClass, typename... Tensors>
    GOpPtr make_op(const std::string &name, OpDescPtr desc, Tensors &&... args) {
        auto op = new OpClass();
        op->set_name(name);
        (*op)(*this, desc, {std::forward<Tensors>(args)...});
        ncg_assert_msg(ok(), error_str());
        auto op_ptr = GOpPtr(op);
        m_ops.push_back(op_ptr);
        return op_ptr;
    }

    const std::vector<GOpPtr> &ops() const;
    GOpPtr find_op(const std::string &name);

    template <typename OpClass, typename... Tensors>
    typename std::enable_if<std::is_base_of<GraphSingleOutputOp, OpClass>::value, GTensorPtr>::type
    op(OpDescPtr desc, Tensors &&... args) {
        auto op = make_op<OpClass>(desc, std::forward<Tensors>(args)...);
        ncg_assert(op->outputs().size() == 1);
        return op->outputs()[0];
    }

    template <typename OpClass, typename... Tensors>
    typename std::enable_if<std::is_base_of<GraphSingleOutputOp, OpClass>::value, GTensorPtr>::type
    op(const std::string &name, OpDescPtr desc, Tensors &&... args) {
        auto op = make_op<OpClass>(name, desc, std::forward<Tensors>(args)...);
        ncg_assert(op->outputs().size() == 1);
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
    std::unordered_set<std::uintptr_t> m_backproped_tensors;
};

class Session {
public:
    Session(Graph &graph);

    Graph &graph();
    const Graph &graph() const;

    bool is_shared_tensor_initialized(const GTensorPtr &) const;
    TensorPtr shared_tensor(const GTensorPtr &) const;
    void set_shared_tensor(const GTensorPtr &gtensor, const TensorPtr &tensor);

protected:
    Graph &m_graph;
    std::unordered_map<std::uintptr_t, TensorPtr> m_shared_tensors;
};

class GraphForwardContext : public OpContext {
public:
    GraphForwardContext();
    GraphForwardContext(Session &session);

    Session &session();
    const Session &session() const;

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

void as_default_graph(Graph &);
Graph &get_default_graph();
void restore_default_graph();

void as_default_session(Session &);
Session &get_default_session();
void restore_default_session();

} /* !namespace ncg */

