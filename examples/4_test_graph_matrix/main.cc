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
    auto x = G::placeholder("x", {3, 4});
    auto y = G::placeholder("y", {4, 5});
    auto bias = as_gtensor<float>({1, 2, 3, 4, 5});
    auto z = G::matmul(x, y) + bias.unsqueeze(0) + float(1);

    cout << z << endl;
    cout << *(z->owner_op()) << endl;

    GraphForwardContext ctx;
    ctx.feed("x", ones(DTypeName::Float32, {3, 4}));
    ctx.feed("y", ones(DTypeName::Float32, {4, 5}));
    auto outputs = ctx.eval({z});

    if (ctx.ok()) {
        cout << outputs[0] << endl;
    } else {
        cerr << ctx.error_str() << endl;
    }

    return 0;
}

