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

    set_shape_vec(shape);
    if (stride.size() > 0) {
        ncg_assert(shape.size() == stride.size());
        set_stride_vec(stride);
    } else {
        set_default_stride();
    }
}

TensorDesc::TensorDesc(NCGUnpickler &unpickler) {
    m_dtype = static_cast<DTypeName>(unpickler.read_int64());
    auto shape = unpickler.read_ssize_array();
    ncg_assert(shape.second == TensorMaxDim + 1);
    memcpy(m_shape, shape.first.get(), sizeof(m_shape));
    auto stride = unpickler.read_ssize_array();
    ncg_assert(stride.second == TensorMaxDim + 1);
    memcpy(m_stride, stride.first.get(), sizeof(m_stride));
}

void TensorDesc::pickle(NCGPickler &pickler) const {
    pickler.write(static_cast<int64_t>(m_dtype));
    pickler.write_ssize_array(m_shape, TensorMaxDim + 1);
    pickler.write_ssize_array(m_stride, TensorMaxDim + 1);
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

void TensorDesc::set_shape_vec(const ShapeVec &shape) {
    ncg_assert(shape.size() <= TensorMaxDim);
    size_t i = 0;
    for (auto it = shape.begin(); it != shape.end(); ++it) m_shape[i++] = *it;
    m_shape[i++] = TensorShape0;
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

void TensorDesc::set_stride_vec(const ShapeVec &stride) {
    ncg_assert(stride.size() <= TensorMaxDim);
    size_t i = 0;
    for (auto it = stride.begin(); it != stride.end(); ++it) m_stride[i++] = *it;
    m_stride[i++] = TensorShape0;
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

bool TensorDesc::is_contiguous() const {
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

 bool TensorDesc::is_scalar_broadcasted() const {
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


size_t TensorDesc::numel() const {
    size_t n = 1;
    for (ssize_t i = 0; i < TensorMaxDim; ++i) {
        if (m_shape[i] == TensorShape0) break;
        n *= m_shape[i];
    }
    return n;
}

bool TensorDesc::is_compatible(const TensorDesc &rhs, bool allow_broadcast) const {
    if (dim() != rhs.dim()) {
        return false;
    }

    for (ssize_t i = 0; i < dim(); ++i) {
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
    out << "TensorDesc(dtype=" << get_dtype_name(desc.m_dtype) << ", dim=" << d << ", shape=[";
    for (ssize_t i = 0; i < d; ++i) out << desc.m_shape[i] << (i == d - 1 ? "" : ", ");
    out << "], stride=[";
    for (ssize_t i = 0; i < d; ++i) out << desc.m_stride[i] << (i == d - 1 ? "" : ", ");
    out << "])";
    return out;
}

} /* !namespace ncg */
