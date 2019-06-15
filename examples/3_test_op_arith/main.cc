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

int main() {
    auto t1 = scalar(DTypeName::Float32, 1);
    auto t2 = scalar(DTypeName::Float32, 2);
    auto t3 = t1 + t2;

    cerr << *t1->as<DTypeName::Float32>() << endl;
    cerr << *t2->as<DTypeName::Float32>() << endl;
    cerr << *t3->as<DTypeName::Float32>() << endl;

    return 0;
}
