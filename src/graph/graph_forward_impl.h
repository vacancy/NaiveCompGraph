/*
 * graph_forward_impl.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#ifndef GRAPH_FORWARD_IMPL_H
#define GRAPH_FORWARD_IMPL_H

#include "graph/graph_op.h"
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

} /* !namespace ncg */

#endif /* !GRAPH_FORWARD_IMPL_H */
