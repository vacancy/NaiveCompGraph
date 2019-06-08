/*
 * op.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#ifndef CORE_OP_H
#define CORE_OP_H

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
    virtual ~Op() = default;

    TensorVec execute(OpContext &ctx, const TensorVec &inputs);
    virtual void check_inputs(OpContext &ctx, const TensorVec &inputs) = 0;
    virtual TensorVec compute(OpContext &ctx, const TensorVec &inputs) = 0;
    virtual const char *op_name() const = 0;

    template <typename DescT>
    const DescT &desc() const {
        auto p = dynamic_cast<DescT *>(m_desc.get());
        ncg_assert(p != nullptr);
        return *p;
    }

    void set_desc(OpDescPtr);

    friend std::ostream & operator << (std::ostream &, const Op &);

protected:
    OpDescPtr m_desc;
};

#define NCG_DEF_OPNAME(op_name_) virtual const char *op_name(void) const { return #op_name_; }

class OpContext {
public:
    OpContext() : m_is_error(false), m_error() {}
    virtual ~OpContext() = default;

    bool ok() const;
    bool is_error() const;

    std::string error_str() const;
    std::ostringstream &error(const Op *);
    void reset_error();

protected:
    bool m_is_error;
    std::ostringstream m_error;
};

} /* !namespace ncg */

#endif /* !CORE_OP_H */
