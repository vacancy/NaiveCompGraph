/*
 * tensor.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include "tensor.h"

namespace ncg {

TensorPtr empty(DTypeName dtype, const std::vector<size_t> &shape) {
    TensorDesc desc(dtype, shape);

#define EMPTY_DTYPE_CASE(dtype) return std::shared_ptr<Tensor>( \
        static_cast<Tensor *>(new TensorImpl<DTypeName::dtype>(\
            desc, new TensorStorage<DTypeName::dtype>(desc.numel()) \
        )) \
    )
NCG_SWITCH_DTYPE_ALL(dtype, EMPTY_DTYPE_CASE)
#undef EMPTY_DTYPE_CASE
    return std::shared_ptr<Tensor>(nullptr);
}


} /* !namespace ncg */
