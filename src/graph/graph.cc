/*
 * graph.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include "graph/tensor.h"
#include "graph/op.h"
#include "graph.h"

namespace ncg {

GraphTopoSorter::GraphTopoSorter() {
    // Pass
}

void GraphTopoSorter::sort(const GTensorVec &target) {
    m_sorted.clear();
    m_visited.clear();

    for (const auto &t : target) {
        mark_(t);
    }
}

const std::vector<const GraphOp *> &GraphTopoSorter::sorted() const {
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

GraphForwardContext::GraphForwardContext(Graph &graph) : m_graph(graph), m_feed_dict(), m_storage() {
    // Pass
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
    auto sorter = std::make_unique<GraphTopoSorter>();
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
    m_is_error = true;
    // m_error << op->name() << ": ";
    return m_error;
}

Graph::Graph() : m_ops(), m_is_error(false), m_error() {
    // Pass
}

bool Graph::ok() const {
    return !m_is_error;
}

bool Graph::is_error() const {
    return m_is_error;
}

std::string Graph::error_str() const {
    return m_error.str();
}

std::ostringstream &Graph::error(const GraphOp *op) {
    m_is_error = true;
    // m_error << op->name() << ": ";
    return m_error;
}

void Graph::backward(GTensorPtr loss) {
    auto sorter = std::make_unique<GraphTopoSorter>();
    sorter->sort({loss});
    const auto &sorted = sorter->sorted();

    for (auto it = sorted.rbegin(); it != sorted.rend(); ++it) {
        (*it)->backward();
    }

    /* TODO:  <02-06-19, Jiayuan Mao> */
}

} /* !namespace ncg */
