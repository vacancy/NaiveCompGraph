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

    auto narrow_op = OpNarrow();
    narrow_op.set_desc(std::shared_ptr<OpDesc>(new OpNarrowDesc(1, 1, 2)));
    auto t3_vec = narrow_op.execute(ctx, t2_vec);
    ncg_assert_msg(!ctx.is_error(), ctx.error_str());
    auto t3 = t3_vec[0];
    cerr << *t3 << endl;
    print_tensor(cerr, t3) << endl;

    auto concat_op = OpConcat();
    concat_op.set_desc(std::shared_ptr<OpDesc>(new OpConcatDesc(0)));
    auto t4_vec = concat_op.execute(ctx, {t3, t3, t3});
    ncg_assert_msg(!ctx.is_error(), ctx.error_str());
    auto t4 = t4_vec[0];
    cerr << *t4 << endl;
    print_tensor(cerr, t4) << endl;

    auto split_op = OpSplit();
    split_op.set_desc(std::shared_ptr<OpDesc>(new OpSplitDesc(0, {1, 4, 4})));
    auto t5_vec = split_op.execute(ctx, t4_vec);
    ncg_assert_msg(!ctx.is_error(), ctx.error_str());

    for (int i = 0; i < 3; ++i) {
        cerr << "t5[" << i << "]:" << endl;
        cerr << *t5_vec[i] << endl;
        print_tensor(cerr, t5_vec[i]) << endl;
    }

    auto index_select_op = OpIndexSelect();
    index_select_op.set_desc(std::shared_ptr<OpDesc>(new OpIndexSelectDesc(0)));
    auto tindex = fromcc(DTypeName::Int64, {0, 2, 2, 2, 4});
    auto t6_vec = index_select_op.execute(ctx, {t4, tindex});
    ncg_assert_msg(!ctx.is_error(), ctx.error_str());
    auto t6 = t6_vec[0];
    cerr << *t6 << endl;
    print_tensor(cerr, t6) << endl;

    auto gather_op = OpGather();
    gather_op.set_desc(std::shared_ptr<OpDesc>(new OpGatherDesc(1)));
    auto tgindex = fromcc(DTypeName::Int64, {0, 1, 1, 1, 0, 1, 1, 1, 0});
    auto reshape_op2 = OpReshape();
    reshape_op2.set_desc(std::shared_ptr<OpDesc>(new OpReshapeDesc({9, 1})));
    auto tgindex_vec = reshape_op2.execute(ctx, {tgindex});
    ncg_assert_msg(!ctx.is_error(), ctx.error_str());
    auto tgindex2 = tgindex_vec[0];
    auto t7_vec = gather_op.execute(ctx, {t4, tgindex2});
    ncg_assert_msg(!ctx.is_error(), ctx.error_str());
    auto t7 = t7_vec[0];
    cerr << *t7 << endl;
    print_tensor(cerr, t7) << endl;

    return 0;
}
