/*
 * op.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#ifndef OP_H
#define OP_H

#include "core/common.h"
#include "core/tensor.h"
#include <string>
#include <sstream>

namespace ncg {

class OpContext;

class Op {
public:
    TensorVec execute(OpContext &, const TensorVec &);
    virtual void check_inputs(OpContext &, const TensorVec &) = 0;
    virtual TensorVec compute(OpContext &, const TensorVec &) = 0;
    virtual const char *name(void) const = 0;
};

#define NCG_DEF_OPNAME(op_name) virtual const char *name(void) const { return #op_name; }

class OpContext {
public:
    OpContext() : m_is_error(false), m_error() {}
    virtual ~OpContext() = default;

    bool ok() const {
        return !m_is_error;
    }

    bool is_error() const {
        return m_is_error;
    }
    std::string error_str() const {
        return m_error.str();
    }
    std::ostringstream &error(const Op *op) {
        m_is_error = true;
        m_error << op->name() << ": ";
        return m_error;
    }
private:
    bool m_is_error;
    std::ostringstream m_error;
};

} /* !namespace ncg */

#endif /* !OP_H */
