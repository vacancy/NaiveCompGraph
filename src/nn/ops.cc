/*
 * ops.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include "nn/ops.h"

#include <string>

namespace ncg {

namespace G {

GTensorPtr linear(std::string name, GTensorPtr x, ssize_t output_dim, std::mt19937 &rng, double stddev) {
    auto W = variable(name + ":W", ::ncg::rand_normal(rng, x->desc().dtype(), {x->desc().shape(1), output_dim}, 0, stddev));
    auto b = variable(name + ":b", ::ncg::zeros(x->desc().dtype(), {output_dim}));
    return matmul(x, W) + b.unsqueeze(0);
}

GTensorPtr softmax(GTensorPtr logits, ssize_t axis) {
    if (axis < 0) axis += logits->desc().dim();

    auto x = logits - logits.max(axis, true)[0];
    auto exp_x = exp(x);
    return exp_x / exp_x.sum(axis, true);
}

GTensorPtr xent_sparse(GTensorPtr probs, GTensorPtr indices, ssize_t axis) {
    if (axis < 0) axis += probs->desc().dim();

    auto mlog_probs = -log(probs);
    return mlog_probs.gather(axis, indices.unsqueeze(axis)).squeeze(axis);
}

}; /* !namespace G */

} /* !namespace ncg */
