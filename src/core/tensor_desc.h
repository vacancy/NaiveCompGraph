/*
 * tensor_desc.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#pragma once

#include "core/common.h"
#include "core/datatype.h"
#include "core/pickle.h"

#include <limits>

namespace ncg {

const ssize_t NoneShape = std::numeric_limits<ssize_t>::min();
const ssize_t NewAxis = std::numeric_limits<ssize_t>::max();

class ShapeVec : public std::vector<ssize_t> {
public:
    using std::vector<ssize_t>::vector;
    virtual ~ShapeVec() = default;
    friend std::ostream &operator << (std::ostream &out, const ShapeVec &shape);
};

class TensorDesc {
/*
 * Tensor description, including dtypes and shapes.
 */
public:
    TensorDesc();
    TensorDesc(DTypeName dtype, const ShapeVec &shape);
    TensorDesc(DTypeName dtype, ShapeVec &&shape);
    virtual ~TensorDesc() = default;

    TensorDesc(const TensorDesc &) = default;
    TensorDesc(TensorDesc &&) = default;

    /* Constructor from an NCGUnpickler. */
    TensorDesc(NCGUnpickler &unpickler);
    /* Dump the data to an NCGPickler . */
    void pickle(NCGPickler &pickler) const;

    /* Return the dtype of the tensor. */
    DTypeName dtype() const;

    /* Return the dimension of the tensor. */
    size_t dim() const;
    /* Return the total number of elments in the tensor. */
    size_t numel() const;

    /* Return the shape (as a vector<ssize_t>) of the tensor. */
    const ShapeVec &shape() const;
    /* Return the size of the i-th dimension of the tensor; const version. */
    ssize_t shape(ssize_t i) const;
    /* Return the size of the i-th dimension of the tensor. */
    ssize_t &shape(ssize_t i);
    /* Set the shape (by a vector<ssize_t>) of the tensor. */
    void set_shape(const ShapeVec &shape);
    void set_shape(ShapeVec &&shape);

    /* Check if two tensor shapes are equal or broadcastable. */
    bool is_shape_compatible(const TensorDesc &rhs, bool allow_broadcast=false) const;
    /* Check if two tensor shapes are equal or broadcastable and the tensor dtypes are equal. */
    bool is_tensor_compatible(const TensorDesc &rhs, bool allow_broadcast=false) const;

    /* Print the tensor description to an ostream. */
    friend std::ostream &operator << (std::ostream &out, const TensorDesc &desc);

protected:
    DTypeName m_dtype;
    ShapeVec m_shape;
};

class TensorLayout : public TensorDesc {
public:
    TensorLayout();
    TensorLayout(const TensorDesc &);
    TensorLayout(TensorDesc &&);

    TensorLayout(DTypeName dtype, const ShapeVec &shape);
    TensorLayout(DTypeName dtype, ShapeVec &&shape);
    TensorLayout(DTypeName dtype, const ShapeVec &shape, const ShapeVec &stride);
    TensorLayout(DTypeName dtype, ShapeVec &&shape, ShapeVec &&stride);
    virtual ~TensorLayout() = default;

    TensorLayout(const TensorLayout &) = default;
    TensorLayout(TensorLayout &&) = default;

    /* Return the stride (as a vector<ssize_t>) of the tensor. */
    const ShapeVec &stride() const;
    /* Return the size of the i-th dimension of the tensor; const version. */
    ssize_t stride(ssize_t i) const;
    /* Return the size of the i-th dimension of the tensor. */
    ssize_t &stride(ssize_t i);
    /* Set the stride (by a vector<ssize_t>) of the tensor. */
    void set_stride(const ShapeVec &stride);
    void set_stride(ShapeVec &&stride);

    /* Make a default stride of the tensor */
    ShapeVec get_default_stride() const;
    /* Use the default stride to overwrite the current stride. */
    void set_default_stride();

    /* Return if the tensor layout is a contiguous layout. */
    bool is_contiguous() const;
    /* Return if the tensor layout is broadcasted from a single scalar. */
    bool is_scalar_broadcasted() const;

    /* Print the tensor description to an ostream. */
    friend std::ostream &operator << (std::ostream &out, const TensorDesc &desc);

protected:
    ShapeVec m_stride;
};

} /* !namespace ncg */

