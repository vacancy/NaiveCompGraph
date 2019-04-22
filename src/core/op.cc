/*
 * op.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include "op.h"

namespace ncg {

TensorVec Op::execute(OpContext &ctx, const TensorVec &inputs) {
    check_inputs(ctx, inputs);
    if (ctx.is_error()) {
        return TensorVec();
    }
    return compute(ctx, inputs);
}

} /* !namespace ncg */
