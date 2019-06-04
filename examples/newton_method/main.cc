/*
 * main.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include "ncg.h"
#include <iostream>
#include <iomanip>

namespace ncg {

} /* !namespace ncg */

using namespace ncg;
using namespace std;

void solve() {
    Graph graph;
    auto x = graph.op<GOpPlaceholder>("x", OpDescPtr(new GOpPlaceholderDesc(DTypeName::Float64, {})));
    auto y = graph.op<GOpConstant>("y0", OpDescPtr(new GOpConstantDesc(scalar(DTypeName::Float64, 0))));

    int k;
    cin >> k;
    for (int i = k; i >= 0; --i) {
        double j;
        cin >> j;

        auto pi = graph.op<GOpConstant>("p" + std::to_string(i), OpDescPtr(new GOpConstantDesc(scalar(DTypeName::Float64, i))));
        auto ai = graph.op<GOpConstant>("a" + std::to_string(i), OpDescPtr(new GOpConstantDesc(scalar(DTypeName::Float64, j))));
        y = graph.op<GOpAdd>(nullptr, y, graph.op<GOpMul>(nullptr, ai, graph.op<GOpPow>(nullptr, x, pi)));
    }

    graph.backward(y);
    auto new_x = graph.op<GOpSub>(nullptr, x, graph.op<GOpDiv>(nullptr, y, x->grad(y)));

    double x_val;
    cin >> x_val;

    for (int i = 0; i < 5; ++i) {
        GraphForwardContext ctx(graph);
        ctx.feed("x", scalar(DTypeName::Float64, x_val));
        auto outputs = ctx.eval({new_x, y, x->grad(y)});

        // cerr << "debug"
        //      << " x = " << x_val
        //      << " ; y = " << outputs[1]->as<DTypeName::Float64>()->elat(0)
        //      << " ; y' = " << outputs[2]->as<DTypeName::Float64>()->elat(0)
        //      << endl;
        x_val = outputs[0]->as<DTypeName::Float64>()->elat(0);
        cout << x_val << " ";
    }
    cout << endl;
}

int main() {
    int k;
    cin >> k;
    cout << fixed << setprecision(4);

    while (k--) {
        solve();
    }

    return 0;
}

