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
    Session session(graph);
    as_default_graph(graph);
    as_default_session(session);

    auto x = G::placeholder("x", {}), y = as_gtensor(float(0));

    int k;
    cin >> k;
    for (int i = k; i >= 0; --i) {
        double j;
        cin >> j;

        float pi = i, ai = j;
        y = y + ai * G::pow(x, pi);
    }

    graph.backward(y);
    auto new_x = x - y / x->grad(y);

    double x_val;
    cin >> x_val;

    for (int i = 0; i < 5; ++i) {
        GraphForwardContext ctx;
        ctx.feed("x", scalar(DTypeName::Float32, x_val));
        auto outputs = ctx.eval({new_x, y, x->grad(y)});

        x_val = outputs[0]->as<DTypeName::Float32>()->elat(0);
        cout << x_val << " ";
    }
    cout << endl;

    restore_default_session();
    restore_default_graph();
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

