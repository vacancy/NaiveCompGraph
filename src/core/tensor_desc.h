/*
 * tensor_desc.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#pragma once

#include "core/common.h"
#include "core/datatype.h"

#include <iostream>
#include <vector>
#include <memory>
#include <limits>

namespace ncg {

const size_t TensorMaxDim = 15;

const ssize_t TensorShape0 = std::numeric_limits<ssize_t>::min();
const ssize_t NoneShape = std::numeric_limits<ssize_t>::min() + 1;
const ssize_t NewAxis = std::numeric_limits<ssize_t>::max();

class ShapeVec : public std::vector<ssize_t> {
public:
    using std::vector<ssize_t>::vector;
    friend std::ostream &operator << (std::ostream &out, const ShapeVec &shape);
};

class TensorDesc {
public:
    TensorDesc();
    TensorDesc(DTypeName dtype, const ShapeVec &shape, const ShapeVec &stride = {});
    virtual ~TensorDesc() = default;

    DTypeName dtype() const;

    size_t dim() const;

    ShapeVec shape_vec() const;
    void set_shape_vec(const ShapeVec &);
    ssize_t *shape();
    const ssize_t *shape() const;
    ShapeVec stride_vec() const;
    void set_stride_vec(const ShapeVec &);
    ssize_t *stride();
    const ssize_t *stride() const;

    ssize_t &shape(ssize_t i);
    ssize_t shape(ssize_t i) const;
    ssize_t &stride(ssize_t i);
    ssize_t stride(ssize_t i) const;

    ShapeVec get_default_stride() const;
    void set_default_stride();

    bool is_continugous() const;
    size_t numel() const;
    bool is_compatible(const TensorDesc &rhs, bool allow_broadcast=false) const;
    friend std::ostream &operator << (std::ostream &out, const TensorDesc &desc);

protected:
    DTypeName m_dtype;
    ssize_t m_shape[TensorMaxDim + 1];
    ssize_t m_stride[TensorMaxDim + 1];
};

class TensorDescVec : public std::vector<TensorDesc> {
public:
    using std::vector<TensorDesc>::vector;
};

} /* !namespace ncg */

