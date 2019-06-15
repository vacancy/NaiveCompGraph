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

void Op::set_desc(OpDescPtr desc) {
    m_desc = desc;
}

std::ostream & operator << (std::ostream &out, const Op &op) {
    return out << op.op_name() << "@" << &op;
}

std::ostringstream &OpContext::error(const Op *op) {
    auto &error = RuntimeContext::error();
    // error << op->op_name() << ": ";
    return error;
}

} /* !namespace ncg */
