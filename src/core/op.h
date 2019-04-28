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
#include <iostream>

namespace ncg {

class OpContext;

class OpDesc {
public:
    virtual ~OpDesc() = default;
};

typedef std::shared_ptr<OpDesc> OpDescPtr;

class Op {
public:
    Op() : m_desc() {}

    TensorVec execute(OpContext &, const TensorVec &);
    virtual void check_inputs(OpContext &, const TensorVec &) = 0;
    virtual TensorVec compute(OpContext &, const TensorVec &) = 0;
    virtual const char *op_name(void) const = 0;

    template <typename DescT>
    const DescT &desc() const {
        return *(dynamic_cast<DescT *>(m_desc.get()));
    }
    void set_desc(OpDescPtr desc) {
        m_desc = desc;
    }

    friend std::ostream & operator << (std::ostream &out, const Op &op) {
        return out << op.op_name() << "@" << &op;
    }

protected:
    OpDescPtr m_desc;
};

#define NCG_DEF_OPNAME(op_name_) virtual const char *op_name(void) const { return #op_name_; }

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
        m_error << op->op_name() << ": ";
        return m_error;
    }
    void reset_error(void) {
        m_is_error = false;
        m_error.clear();
    }

protected:
    bool m_is_error;
    std::ostringstream m_error;
};

} /* !namespace ncg */

#endif /* !OP_H */
