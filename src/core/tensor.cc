/*
 * tensor.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include "tensor.h"

namespace ncg {

#define EMPTY_DTYPE_CASE(dtype) case DTypeName::dtype: \
	return std::shared_ptr<Tensor>( \
        static_cast<Tensor *>(new TensorImpl<DTypeName::dtype>(\
            desc, new TensorStorage<DTypeName::dtype>(desc.numel()) \
        )) \
    )

TensorPtr empty(DTypeName dtype, const std::initializer_list<size_t> &shape) {
	TensorDesc desc(dtype, shape);

	switch (dtype) {
		EMPTY_DTYPE_CASE(Int8);
		EMPTY_DTYPE_CASE(UInt8);
		EMPTY_DTYPE_CASE(Int32);
		EMPTY_DTYPE_CASE(UInt32);
		EMPTY_DTYPE_CASE(Int64);
		EMPTY_DTYPE_CASE(UInt64);
		EMPTY_DTYPE_CASE(Float32);
		EMPTY_DTYPE_CASE(Float64);
	}

    return std::shared_ptr<Tensor>(nullptr);
}

#undef EMPTY_DTYPE_CASE

} /* !namespace ncg */
