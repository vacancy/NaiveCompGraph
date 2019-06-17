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
    auto x = G::placeholder("x", {}, DTypeName::Float32);
    auto y = G::placeholder("y", {}, DTypeName::Float32);
    auto z = x + y;

    cout << z << endl;
    cout << *(z->owner_op()) << endl;

    GraphForwardContext ctx;
    ctx.feed("x", scalar(DTypeName::Float32, 1));
    ctx.feed("y", scalar(DTypeName::Float32, 2));
    auto outputs = ctx.eval({z});

    if (ctx.ok()) {
        cout << outputs[0] << endl;
    } else {
        cerr << ctx.error_str() << endl;
    }

    return 0;
}

