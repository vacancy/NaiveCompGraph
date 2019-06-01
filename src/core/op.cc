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


bool OpContext::ok() const {
    return !m_is_error;
}

bool OpContext::is_error() const {
    return m_is_error;
}
std::string OpContext::error_str() const {
    return m_error.str();
}
std::ostringstream &OpContext::error(const Op *op) {
    m_is_error = true;
    // m_error << op->op_name() << ": ";
    return m_error;
}
void OpContext::reset_error() {
    m_is_error = false;
    m_error.clear();
}


} /* !namespace ncg */
