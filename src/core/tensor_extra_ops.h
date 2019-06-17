/*
 * tensor_extra_ops.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#pragma once

#include "tensor.h"
#include <random>

namespace ncg {

using URBG = std::mt19937;

TensorPtr rand_uniform(URBG& rng, DTypeName dtype, const ShapeVec &shape, double low = 0.0, double high = 1.0);
TensorPtr rand_normal(URBG& rng, DTypeName dtype, const ShapeVec &shape, double mean = 0.0, double stddev = 1.0);
TensorPtr rand_permutation(URBG& rng, ssize_t size);

} /* !namespace ncg */

