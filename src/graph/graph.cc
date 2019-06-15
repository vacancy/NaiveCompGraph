/*
 * graph.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include "graph/tensor.h"
#include "graph/op.h"
#include "graph/graph.h"
#include "graph/ops/grad.h"

namespace ncg {

GraphTopoSorter::GraphTopoSorter(Graph &graph) : m_graph(graph) {
    // Pass
}

void GraphTopoSorter::sort(const GTensorVec &target) {
    m_sorted.clear();
    m_visited.clear();

    for (const auto &t : target) {
        mark_(t);
    }
}

const std::vector<GraphOp *> &GraphTopoSorter::sorted() const {
    return m_sorted;
}

void GraphTopoSorter::mark_(const GTensorPtr &t) {
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

Graph::Graph() : m_ops() {
    // Pass
}

std::ostringstream &Graph::error(const GraphOp *op) {
    auto &error = RuntimeContext::error();
    // error << op->op_name() << ": ";
    return error;
}

GOpPtr Graph::find_op(const std::string &name) {
    for (auto &op : m_ops) {
        if (op->name() == name) {
            return op;
        }
    }
    return nullptr;
}

void Graph::backward(GTensorPtr loss) {
    auto sorter = std::make_unique<GraphTopoSorter>(*this);
    sorter->sort({loss});
    const auto &sorted = sorter->sorted();

    loss->set_grad(*this, loss, this->op<GOpGradLoss>(nullptr, loss));
    for (auto it = sorted.rbegin(); it != sorted.rend(); ++it) {
        (*it)->backward(*this, loss);
    }
}

Session::Session(Graph &graph) : m_graph(graph) {
    // pass
}

Graph &Session::Session::graph() {
    return m_graph;
}

const Graph &Session::graph() const {
    return m_graph;
}

bool Session::is_shared_tensor_initialized(const GTensorPtr &gtensor) const {
    auto it = m_shared_tensors.find(reinterpret_cast<std::uintptr_t>(gtensor.get()));
    return it != m_shared_tensors.end();
}

TensorPtr Session::shared_tensor(const GTensorPtr &gtensor) const {
    auto it = m_shared_tensors.find(reinterpret_cast<std::uintptr_t>(gtensor.get()));
    ncg_assert(it != m_shared_tensors.end());
    return it->second;
}

void Session::set_shared_tensor(const GTensorPtr &gtensor, const TensorPtr &tensor) {
    m_shared_tensors[reinterpret_cast<std::uintptr_t>(gtensor.get())] = tensor;
}

GraphForwardContext::GraphForwardContext(Session &session) : m_session(session), m_feed_dict(), m_storage() {
    // Pass
}

Session &GraphForwardContext::session() {
    return m_session;
}

const Session &GraphForwardContext::session() const {
    return m_session;
}

void GraphForwardContext::feed(const std::string &name, TensorPtr tensor) {
    m_feed_dict.emplace(name, tensor);
}

TensorPtr GraphForwardContext::feed_dict(const std::string &name) {
    auto it = m_feed_dict.find(name);
    if (it == m_feed_dict.end()) {
        return nullptr;
    }
    return it->second;
}

TensorVec GraphForwardContext::eval(const GTensorVec &targets) {
    TensorVec outputs;
    auto sorter = std::make_unique<GraphTopoSorter>(m_session.graph());
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

TensorPtr GraphForwardContext::tensor(const GTensorPtr &gtensor) {
    auto it = m_storage.find(reinterpret_cast<std::uintptr_t>(gtensor.get()));
    ncg_assert(it != m_storage.end());
    return it->second;
}

void GraphForwardContext::set_tensor(const GTensorPtr &gtensor, const TensorPtr &tensor) {
    m_storage.emplace(reinterpret_cast<std::uintptr_t>(gtensor.get()), tensor);
}

std::ostringstream &GraphForwardContext::error(const GraphOp *op) {
    auto &error = RuntimeContext::error();
    // error << op->op_name() << ": ";
    return error;
}

} /* !namespace ncg */
