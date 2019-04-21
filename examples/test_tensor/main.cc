/*
 * main.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include "core/tensor.h"
#include <iostream>

namespace ncg {

} /* !namespace ncg */

using namespace ncg;
using namespace std;

int main() {
    auto t1 = empty(DTypeName::Float32, {3, 3});
    cerr << t1->as<DTypeName::Float32>() << endl;

    auto t2 = empty(DTypeName::Float32, {});
    cerr << t2->as<DTypeName::Float32>() << endl;

    return 0;
}
