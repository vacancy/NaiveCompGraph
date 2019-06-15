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
    auto t3 = t2.permute({1, 0});
    P(t3);
    auto t4 = t3.reshape({4, 3, 1});
    P(t4);
    auto t5 = t4.expand({4, 3, 2});
    P(t5);
    auto t6 = t5.reshape({4, 6});
    P(t6);

    return 0;
}
