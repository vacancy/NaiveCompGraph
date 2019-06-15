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
#define P(tensor) do { \
    cerr << #tensor << ":" << tensor << endl; \
    if (tensor->desc().dim() == 2) { \
        print_tensor(cerr, tensor) << endl; \
    } \
} while(0)

    auto t1 = arange(DTypeName::Float32, 12);
    P(t1);
    auto t2 = t1.reshape({3, 4});
    P(t2);
    auto t3 = t2.narrow(1, 1, 2);
    P(t3);
    auto t4 = concat({t3, t3, t3}, 0);
    P(t4);
    auto t5 = split(t4, 0, {1, 4, 4});
    P(t5[0]); P(t5[1]); P(t5[2]);

    auto tindex = as_tensor<int64_t>({0, 2, 2, 2, 4});
    auto t6 = t4.index_select(0, tindex);
    P(t6);

    auto tgindex = fromcc(DTypeName::Int64, {0, 1, 1, 1, 0, 1, 1, 1, 0});
    auto t7 = t4.gather(1, tgindex.reshape({9, 1}));
    P(t7);

    return 0;
}
