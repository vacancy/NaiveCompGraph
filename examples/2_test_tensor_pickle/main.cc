/*
 * main.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include "core/tensor.h"
#include "core/tensor_impl.h"
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
    auto t1 = arange(DTypeName::Float32, 18).reshape({9, 2});
    print_tensor(cerr << "t1:" << std::endl, t1) << std::endl;

    NCGPickler pkl("test.bin");
    t1->pickle(pkl);
    pkl.close();
    NCGUnpickler unpkl("test.bin");
    auto t2 = tensor(unpkl);
    unpkl.close();

    print_tensor(cerr << "t2:" << std::endl, t2) << std::endl;

    return 0;
}
