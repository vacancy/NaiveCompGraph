/*
 * main.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include "core/tensor.h"
#include "core/op.h"
#include "ops/arith.h"
#include <iostream>

namespace ncg {

} /* !namespace ncg */

using namespace ncg;
using namespace std;

int main() {
    auto t1 = scalar(DTypeName::Float32, 1);
    auto t2 = scalar(DTypeName::Float32, 2);

    auto add_op = OpAdd();
    auto ctx = OpContext();
    auto t3_vec = add_op.execute(ctx, {t1, t2});
    if (ctx.is_error()) {
        ncg_assert_msg(!ctx.is_error(), ctx.error_str());
    }
    auto t3 = t3_vec[0];

    cerr << *t1->as<DTypeName::Float32>() << endl;
    cerr << *t2->as<DTypeName::Float32>() << endl;
    cerr << *t3->as<DTypeName::Float32>() << endl;

    return 0;
}
