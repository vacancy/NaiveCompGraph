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
    auto y = graph.op<GOpPlaceholder>("y", OpDescPtr(new GOpPlaceholderDesc(DTypeName::Float32, {})));
    auto z = graph.op<GOpAdd>(nullptr, x, y);

    cout << *z << endl;
    cout << *(z->owner_op()) << endl;

    Session session(graph);
    GraphForwardContext ctx(session);
    ctx.feed("x", scalar(DTypeName::Float32, 1));
    ctx.feed("y", scalar(DTypeName::Float32, 2));
    auto outputs = ctx.eval({z});

    if (ctx.ok()) {
        cout << *(outputs[0]->as<DTypeName::Float32>()) << endl;
    } else {
        cerr << ctx.error_str() << endl;
    }

    return 0;
}
