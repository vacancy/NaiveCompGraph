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
    auto x = graph.op<GOpPlaceholder>("x", OpDescPtr(new GOpPlaceholderDesc(DTypeName::Float32, {})));
    auto z = graph.op<GOpMul>(nullptr, x, x);
    graph.backward(z);
    auto gx = x->grad(z);

    if (!graph.ok()) {
        cerr << graph.error_str() << endl;
        return 0;
    }

    cout << *gx << endl;
    cout << *(gx->owner_op()) << endl;

    GraphForwardContext ctx(graph);
    ctx.feed("x", scalar(DTypeName::Float32, 2));
    auto outputs = ctx.eval({z, gx});

    if (ctx.ok()) {
        cout << *(outputs[0]->as<DTypeName::Float32>()) << endl;
        cout << *(outputs[1]->as<DTypeName::Float32>()) << endl;
    } else {
        cerr << ctx.error_str() << endl;
    }

    return 0;
}

