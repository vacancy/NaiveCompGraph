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
    auto y = graph.op<GOpConstant>("y", OpDescPtr(new GOpConstantDesc(scalar(DTypeName::Float32, 2))));

    auto z1 = graph.op<GOpAdd>(nullptr, x, x);
    auto z2 = graph.op<GOpMul>(nullptr, x, x);
    auto z3 = graph.op<GOpPow>(nullptr, x, y);

    graph.backward(z1);
    graph.backward(z2);
    graph.backward(z3);
    auto gx1 = x->grad(z1);
    auto gx2 = x->grad(z2);
    auto gx3 = x->grad(z3);

    if (!graph.ok()) {
        cerr << graph.error_str() << endl;
        return 0;
    }

    cout << *gx1 << ", " << *gx2 << ", " << *gx3 << endl;

    Session session(graph);
    GraphForwardContext ctx(session);
    ctx.feed("x", scalar(DTypeName::Float32, 3));
    auto outputs = ctx.eval({gx1, gx2, gx3});

    if (ctx.ok()) {
        cout << *(outputs[0]->as<DTypeName::Float32>()) << endl;
        cout << *(outputs[1]->as<DTypeName::Float32>()) << endl;
        cout << *(outputs[2]->as<DTypeName::Float32>()) << endl;
    } else {
        cerr << ctx.error_str() << endl;
    }

    return 0;
}

