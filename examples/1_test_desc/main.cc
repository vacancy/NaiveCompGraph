/*
 * main.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include "core/common.h"
#include "core/datatype.h"
#include "core/tensor_desc.h"
#include <iostream>
#include <typeinfo>

namespace ncg {

} /* !namespace ncg */

template <ncg::DTypeName DT>
typename ncg::DType<DT>::cctype f() {
    return static_cast<typename ncg::DType<DT>::cctype>(1.345);
}

int main() {
    std::cout << ncg::DType<ncg::DTypeName::Float32>::name << " " << f<ncg::DTypeName::Float32>() << std::endl;
    std::cout << ncg::DType<ncg::DTypeName::Int32>::name << " " << f<ncg::DTypeName::Int32>() << std::endl;
    std::cout << ncg::CCType<float>::name << std::endl;
    std::cout << ncg::CCType<int32_t>::name << std::endl;

    return 0;
}
