/*
 * main.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include "core.h"
#include <iostream>

namespace ncg {

} /* !namespace ncg */

using namespace ncg;
using namespace std;

ostream &print_tensor(ostream &out, TensorPtr tensor) {
    ncg_assert(tensor->desc().dim() == 2);
    out << "[";
    for (ssize_t i = 0; i < tensor->desc().shape(0); ++i) {
        if (i != 0) out << endl << " ";
        out << "[";
        for (ssize_t j = 0; j < tensor->desc().shape(1); ++j) {
            if (j != 0) out << ", ";
            out << tensor->as<DTypeName::Float32>()->at(i, j);
        }
        out << "],";
    }

    return out;
}

int main() {
    auto t1 = arange(DTypeName::Float32, 12);
    cerr << *t1 << endl;

    auto ctx = OpContext();

    auto reshape_op = OpReshape();
    reshape_op.set_desc(std::shared_ptr<OpDesc>(new OpReshapeDesc({3, 4})));
    auto t2_vec = reshape_op.execute(ctx, {t1});
    ncg_assert_msg(!ctx.is_error(), ctx.error_str());
    auto t2 = t2_vec[0];
    cerr << *t2 << endl;
    print_tensor(cerr, t2) << endl;

    auto permute_op = OpPermute();
    permute_op.set_desc(std::shared_ptr<OpDesc>(new OpPermuteDesc({1, 0})));
    auto t3_vec = permute_op.execute(ctx, t2_vec);
    ncg_assert_msg(!ctx.is_error(), ctx.error_str());
    auto t3 = t3_vec[0];
    cerr << *t3 << endl;
    print_tensor(cerr, t3) << endl;

    auto reshape_op2 = OpReshape();
    reshape_op2.set_desc(std::shared_ptr<OpDesc>(new OpReshapeDesc({4, 3, 1})));
    auto t4_vec = reshape_op2.execute(ctx, t3_vec);
    ncg_assert_msg(!ctx.is_error(), ctx.error_str());
    auto t4 = t4_vec[0];
    cerr << *t4 << endl;

    auto expand_op = OpExpand();
    expand_op.set_desc(std::shared_ptr<OpDesc>(new OpExpandDesc({4, 3, 2})));
    auto t5_vec = expand_op.execute(ctx, t4_vec);
    ncg_assert_msg(!ctx.is_error(), ctx.error_str());
    auto t5 = t5_vec[0];
    cerr << *t5 << endl;

    auto reshape_op3 = OpReshape();
    reshape_op3.set_desc(std::shared_ptr<OpDesc>(new OpReshapeDesc({4, 6})));
    auto t6_vec = reshape_op3.execute(ctx, t5_vec);
    ncg_assert_msg(!ctx.is_error(), ctx.error_str());
    auto t6 = t6_vec[0];
    cerr << *t6 << endl;
    print_tensor(cerr, t6) << endl;

    return 0;
}
