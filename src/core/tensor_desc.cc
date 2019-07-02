/*
 * tensor_desc.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include "core/tensor_desc.h"

#include <cstring>

namespace ncg {

std::ostream &operator << (std::ostream &out, const ShapeVec &shape) {
    out << "[";
    for (ssize_t i = 0; i < shape.size() ; ++i) {
        if (i != 0) out << ", ";
        out << shape[i];
    }
    out << "]";
    return out;
}

TensorDesc::TensorDesc() : m_dtype(DTypeName::UInt8), m_shape() {}
TensorDesc::TensorDesc(DTypeName dtype, const ShapeVec &shape) : m_dtype(dtype), m_shape(shape) {}
TensorDesc::TensorDesc(DTypeName dtype, ShapeVec &&shape) : m_dtype(dtype), m_shape(shape) {}

TensorDesc::TensorDesc(NCGUnpickler &unpickler) {
    m_dtype = static_cast<DTypeName>(unpickler.read_int64());
    auto shape = unpickler.read_ssize_array();
    m_shape = ShapeVec(shape.first.get(), shape.first.get() + shape.second);
}

void TensorDesc::pickle(NCGPickler &pickler) const {
    pickler.write(static_cast<int64_t>(m_dtype));
    pickler.write_ssize_array(&m_shape[0], m_shape.size());
}

DTypeName TensorDesc::dtype() const {
    return m_dtype;
}

size_t TensorDesc::dim() const {
    return m_shape.size();
}

size_t TensorDesc::numel() const {
    size_t n = 1;
    const size_t d = dim();
    for (ssize_t i = 0; i < d; ++i) {
        n *= m_shape[i];
    }
    return n;
}

const ShapeVec &TensorDesc::shape() const {
    return m_shape;
}
ssize_t TensorDesc::shape(ssize_t i) const {
    return m_shape[i];
}
ssize_t &TensorDesc::shape(ssize_t i) {
    return m_shape[i];
}
void TensorDesc::set_shape(const ShapeVec &shape) {
    m_shape = shape;
}
void TensorDesc::set_shape(ShapeVec &&shape) {
    m_shape = shape;
}

bool TensorDesc::is_shape_compatible(const TensorDesc &rhs, bool allow_broadcast) const {
    const size_t d = dim();

    if (d != rhs.dim()) {
        return false;
    }

    for (ssize_t i = 0; i < d; ++i) {
        if (allow_broadcast) {
            if (m_shape[i] != rhs.m_shape[i] && !(m_shape[i] == 1 || rhs.m_shape[i] == 1)) return false;
        } else {
            if (m_shape[i] != rhs.m_shape[i]) return false;
        }
    }
    return true;
}

bool TensorDesc::is_tensor_compatible(const TensorDesc &rhs, bool allow_broadcast) const {
    return is_shape_compatible(rhs, allow_broadcast) && dtype() == rhs.dtype();
}

std::ostream &operator << (std::ostream &out, const TensorDesc &desc) {
    const size_t d = desc.dim();
    out << "TensorDesc(";
        out << "dtype=" << get_dtype_name(desc.m_dtype) << ", ";
        out << "dim=" << d << ", ";
        out << "shape=" << desc.m_shape;
    out << ")";
    return out;
}

TensorLayout::TensorLayout() : TensorDesc(), m_stride() {}
TensorLayout::TensorLayout(const TensorDesc &desc) : TensorDesc(desc) {
    set_default_stride();
}
TensorLayout::TensorLayout(TensorDesc &&desc) : TensorDesc(desc) {
    set_default_stride();
}
TensorLayout::TensorLayout(DTypeName dtype, const ShapeVec &shape) : TensorDesc(dtype, shape) {
    set_default_stride();
}
TensorLayout::TensorLayout(DTypeName dtype, ShapeVec &&shape) : TensorDesc(dtype, shape) {
    set_default_stride();
}
TensorLayout::TensorLayout(DTypeName dtype, const ShapeVec &shape, const ShapeVec &stride) : TensorDesc(dtype, shape), m_stride(stride) {
    ncg_assert(shape.size() == stride.size());
}
TensorLayout::TensorLayout(DTypeName dtype, ShapeVec &&shape, ShapeVec &&stride) : TensorDesc(dtype, shape), m_stride(stride) {
    ncg_assert(shape.size() == stride.size());
}

const ShapeVec &TensorLayout::stride() const {
    return m_stride;
}
ssize_t TensorLayout::stride(ssize_t i) const {
    return m_stride[i];
}
ssize_t &TensorLayout::stride(ssize_t i) {
    return m_stride[i];
}
void TensorLayout::set_stride(const ShapeVec &stride) {
    m_stride = stride;
}
void TensorLayout::set_stride(ShapeVec &&stride) {
    m_stride = stride;
}

ShapeVec TensorLayout::get_default_stride() const {
    const size_t d = this->dim();
    ShapeVec stride_vec(d);
    if (d == 0) {
        // pass
    } else {
        stride_vec[d - 1] = 1;
        for (ssize_t i = d - 2; i >= 0; --i) {
            stride_vec[i] = stride_vec[i + 1] * m_shape[i + 1];
        }
    }
    return stride_vec;
}

void TensorLayout::set_default_stride() {
    set_stride(get_default_stride());
}

bool TensorLayout::is_contiguous() const {
    const size_t d = dim();
    if (d == 0) {
        return true;
    }
    if (m_stride[d - 1] != 1) return false;
    for (ssize_t i = d - 2; i >= 0; --i) {
        if (m_stride[i] != m_stride[i + 1] * m_shape[i + 1])
            return false;
    }
    return true;
}

 bool TensorLayout::is_scalar_broadcasted() const {
    size_t d = dim();
    if (d == 0) {
        return true;
    }
    for (ssize_t i = 0; i < d; ++i) {
        if (m_shape[i] > 1 && m_stride[i] != 0) {
            return false;
        }
    }
    return true;
}

} /* !namespace ncg */
