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
    auto tensor_val = (tensor); \
    cerr << #tensor << ":" << tensor_val << endl; \
    if (tensor_val->desc().dim() == 2) { \
        print_tensor(cerr, tensor_val) << endl; \
    } \
} while(0)

    auto t1 = arange(DTypeName::Float32, 24);
    P(t1);
    auto t2 = t1.reshape({2, 3, 4});
    P(t2);

    P(t2.min(2)[0]);
    P(t2.max(-1)[0]);
    P(t2.sum(2));
    P(t2.mean(-1));

    return 0;
}
