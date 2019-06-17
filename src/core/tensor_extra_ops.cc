/*
 * tensor_extra_ops.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include "core/tensor_extra_ops.h"

namespace ncg {

TensorPtr rand_uniform(URBG& rng, DTypeName dtype, const ShapeVec &shape, double low, double high) {
    ncg_assert(dtype == DTypeName::Float32 || dtype == DTypeName::Float64);
    auto s = empty(dtype, shape);

#define RAND_DTYPE_CASE(dtype_name) do { \
    auto s_dtype = s->template as<DTypeName::dtype_name>();\
    std::uniform_real_distribution<typename DType<DTypeName::dtype_name>::cctype> dis(low, high); \
    for (ssize_t i = 0; i < s->desc().numel(); ++i) { s_dtype->mutable_elat(i) = dis(rng); } \
} while(0)
NCG_DTYPE_SWITCH_FLOAT(dtype, RAND_DTYPE_CASE);
#undef RAND_DTYPE_CASE

    return s;
}

TensorPtr rand_normal(URBG& rng, DTypeName dtype, const ShapeVec &shape, double mean, double stddev) {
    ncg_assert(dtype == DTypeName::Float32 || dtype == DTypeName::Float64);
    auto s = empty(dtype, shape);

#define RAND_DTYPE_CASE(dtype_name) do { \
    auto s_dtype = s->template as<DTypeName::dtype_name>();\
    std::normal_distribution<typename DType<DTypeName::dtype_name>::cctype> dis(mean, stddev); \
    for (ssize_t i = 0; i < s->desc().numel(); ++i) { s_dtype->mutable_elat(i) = dis(rng); } \
} while(0)
NCG_DTYPE_SWITCH_FLOAT(dtype, RAND_DTYPE_CASE);
#undef RAND_DTYPE_CASE

    return s;
}

TensorPtr rand_permutation(URBG& rng, ssize_t size) {
    auto s = empty(DTypeName::Int64, {size});
    auto s_dtype = s->template as<DTypeName::Int64>();
    for (ssize_t i = 0; i < size; ++i) { s_dtype->mutable_elat(i) = static_cast<DType<DTypeName::Int64>::cctype>(i); }
    std::shuffle(s_dtype->mutable_data_ptr(), s_dtype->mutable_data_ptr() + size, rng);
    return s;
}

} /* !namespace ncg */
