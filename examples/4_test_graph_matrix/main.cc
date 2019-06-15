/*
 * main.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include "ncg.h"
#include <iostream>

namespace ncg {

} /* !namespace ncg */

using namespace ncg;
using namespace std;

int main() {
    Graph graph;
    auto x = graph.op<GOpPlaceholder>("x", OpDescPtr(new GOpPlaceholderDesc(DTypeName::Float32, {3, 4})));
    auto y = graph.op<GOpPlaceholder>("y", OpDescPtr(new GOpPlaceholderDesc(DTypeName::Float32, {4, 5})));
    auto z = graph.op<GOpMatMul>(OpDescPtr(new OpMatMulDesc()), x, y);

    cout << *z << endl;
    cout << *(z->owner_op()) << endl;

    Session session(graph);
    GraphForwardContext ctx(session);
    ctx.feed("x", ones(DTypeName::Float32, {3, 4}));
    ctx.feed("y", ones(DTypeName::Float32, {4, 5}));
    auto outputs = ctx.eval({z});

    if (ctx.ok()) {
        cout << *(outputs[0]) << endl;
    } else {
        cerr << ctx.error_str() << endl;
    }

    return 0;
}

