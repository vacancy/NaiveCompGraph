/*
 * ops.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#pragma once

#include "core.h"
#include "graph.h"
#include <random>

namespace ncg {

namespace G {

GTensorPtr linear(std::string name, GTensorPtr x, ssize_t output_dim, std::mt19937 &rng, double stddev=0.01);
GTensorPtr softmax(GTensorPtr logits, ssize_t axis);
GTensorPtr xent_sparse(GTensorPtr probs, GTensorPtr indices, ssize_t axis);

}; /* !namespace G */

} /* !namespace ncg */

