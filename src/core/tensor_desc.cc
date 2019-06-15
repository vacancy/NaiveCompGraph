/*
 * tensor_desc.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include "core/tensor_desc.h"

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

TensorDesc::TensorDesc() {
    m_dtype = DTypeName::UInt8;
    memset(m_shape, 0, sizeof(m_shape));
    memset(m_stride, 0, sizeof(m_stride));
}

TensorDesc::TensorDesc(DTypeName dtype, const ShapeVec &shape, const ShapeVec &stride) : m_dtype(dtype) {
    memset(m_shape, 0, sizeof(m_shape));
    memset(m_stride, 0, sizeof(m_stride));

    ncg_assert(shape.size() <= TensorMaxDim);
    ncg_assert(stride.size() <= TensorMaxDim);
    size_t i;

    i = 0;
    for (auto it = shape.begin(); it != shape.end(); ++it) m_shape[i++] = *it;
    m_shape[i++] = TensorShape0;
    size_t d = i - 1;

    if (stride.size() > 0) {
        ncg_assert(shape.size() == stride.size());
        i = 0;
        for (auto it = stride.begin(); it != stride.end(); ++it) m_stride[i++] = *it;
        m_stride[i++] = TensorShape0;
    } else {
        set_default_stride();
    }
}

DTypeName TensorDesc::dtype() const {
    return m_dtype;
}

size_t TensorDesc::dim() const {
    size_t i;
    for (i = 0; i <= TensorMaxDim; ++i)
        if (m_shape[i] == TensorShape0) break;
    return i;
}

ShapeVec TensorDesc::shape_vec() const {
    return ShapeVec(m_shape, m_shape + dim());
}

ssize_t *TensorDesc::shape() {
    return m_shape;
}

const ssize_t *TensorDesc::shape() const {
    return m_shape;
}

ShapeVec TensorDesc::stride_vec() const {
    return ShapeVec(m_stride, m_stride + dim());
}

ssize_t *TensorDesc::stride() {
    return m_stride;
}

const ssize_t *TensorDesc::stride() const {
    return m_stride;
}

ssize_t &TensorDesc::shape(ssize_t i) {
    return m_shape[i];
}

ssize_t TensorDesc::shape(ssize_t i) const {
    return m_shape[i];
}

ssize_t &TensorDesc::stride(ssize_t i) {
    return m_stride[i];
}

ssize_t TensorDesc::stride(ssize_t i) const {
    return m_stride[i];
}

ShapeVec TensorDesc::get_default_stride() const {
    size_t d = dim();
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

void TensorDesc::set_default_stride() {
    size_t d = dim();

    memset(m_stride, 0, sizeof(m_stride));
    if (d == 0) {
        // pass
    } else {
        m_stride[d - 1] = 1;
        for (ssize_t i = d - 2; i >= 0; --i) {
            m_stride[i] = m_stride[i + 1] * m_shape[i + 1];
        }
    }
}

bool TensorDesc::is_continugous() const {
    size_t d = dim();
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

size_t TensorDesc::numel() const {
    size_t n = 1;
    for (ssize_t i = 0; i < TensorMaxDim; ++i) {
        if (m_shape[i] == TensorShape0) break;
        n *= m_shape[i];
    }
    return n;
}

bool TensorDesc::is_compatible(const TensorDesc &rhs, bool allow_broadcast) const {
    for (ssize_t i = 0; i < TensorMaxDim; ++i) {
        if (allow_broadcast) {
            if (m_shape[i] != rhs.m_shape[i] && !(m_shape[i] == 1 || rhs.m_shape[i] == 1)) return false;
        } else {
            if (m_shape[i] != rhs.m_shape[i]) return false;
        }
    }
    return true;
}

std::ostream &operator << (std::ostream &out, const TensorDesc &desc) {
    size_t d = desc.dim();
    out << "TensorDesc(" << "dim=" << d << ", shape=[";
    for (ssize_t i = 0; i < d; ++i) out << desc.m_shape[i] << (i == d - 1 ? "" : ", ");
    out << "], stride=[";
    for (ssize_t i = 0; i < d; ++i) out << desc.m_stride[i] << (i == d - 1 ? "" : ", ");
    out << "])";
    return out;
}

} /* !namespace ncg */
