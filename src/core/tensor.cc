/*
 * tensor.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include "tensor.h"

namespace ncg {

TensorDesc::TensorDesc() {
    m_dtype = DTypeName::UInt8;
    memset(m_shape, 0, sizeof(m_shape));
    memset(m_stride, 0, sizeof(m_stride));
}

TensorDesc::TensorDesc(DTypeName dtype, const std::vector<size_t> &shape, const std::vector<size_t> &stride = {}) : m_dtype(dtype) {
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
        if (d == 0) {
            // pass
        } else {
            m_stride[d - 1] = 1;
            for (ssize_t i = d - 2; i >= 0; --i) {
                m_stride[i] = m_stride[i + 1] * m_shape[i + 1];
            }
        }
    }
}

DTypeName TensorDesc::dtype() const {
    return m_dtype;
}

size_t TensorDesc::dim(void) const {
    size_t i;
    for (i = 0; i <= TensorMaxDim; ++i)
        if (m_shape[i] == TensorShape0) break;
    return i;
}

std::vector<size_t> TensorDesc::shape_vec(void) const {
    return std::vector<size_t>(m_shape, m_shape + dim());
}

size_t *TensorDesc::shape(void) { return m_shape; }
const size_t *TensorDesc::shape(void) const { return m_shape; }
size_t *TensorDesc::stride(void) { return m_stride; }
const size_t *TensorDesc::stride(void) const { return m_stride; }

size_t &TensorDesc::shape(ssize_t i) { return m_shape[i]; }
size_t TensorDesc::shape(ssize_t i) const { return m_shape[i]; }
size_t &TensorDesc::stride(ssize_t i) { return m_stride[i]; }
size_t TensorDesc::stride(ssize_t i) const { return m_stride[i]; }

bool TensorDesc::is_continugous() {
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
        if (m_shape[i] == -1) break;
        n *= m_shape[i];
    }
    return n;
}

bool TensorDesc::is_compatible(const TensorDesc &rhs) {
    if (m_dtype != rhs.m_dtype) return false;
    for (ssize_t i = 0; i < TensorMaxDim; ++i) {
        if (m_shape[i] != rhs.m_shape[i]) return false;
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

TensorPtr empty(DTypeName dtype, const std::vector<size_t> &shape) {
    TensorDesc desc(dtype, shape);

#define EMPTY_DTYPE_CASE(dtype) return std::shared_ptr<Tensor>( \
        static_cast<Tensor *>(new TensorImpl<DTypeName::dtype>(\
            desc, new TensorStorage<DTypeName::dtype>(desc.numel()) \
        )) \
    )
NCG_SWITCH_DTYPE_ALL(dtype, EMPTY_DTYPE_CASE)
#undef EMPTY_DTYPE_CASE
    return std::shared_ptr<Tensor>(nullptr);
}


} /* !namespace ncg */
