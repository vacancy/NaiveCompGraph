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

TensorDesc::TensorDesc(DTypeName dtype, const std::vector<size_t> &shape, const std::vector<size_t> &stride) : m_dtype(dtype) {
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

size_t *TensorDesc::shape(void) {
    return m_shape;
}

const size_t *TensorDesc::shape(void) const {
    return m_shape;
}

size_t *TensorDesc::stride(void) {
    return m_stride;
}

const size_t *TensorDesc::stride(void) const {
    return m_stride;
}

size_t &TensorDesc::shape(ssize_t i) {
    return m_shape[i];
}

size_t TensorDesc::shape(ssize_t i) const {
    return m_shape[i];
}

size_t &TensorDesc::stride(ssize_t i) {
    return m_stride[i];
}

size_t TensorDesc::stride(ssize_t i) const {
    return m_stride[i];
}

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

template <DTypeName DT>
TensorStorage<DT>::TensorStorage() : m_data_ptr(nullptr) {

}

template <DTypeName DT>
TensorStorage<DT>::TensorStorage(cctype *data_ptr, size_t size) : m_data_ptr(data_ptr), m_size(size) {

}

template <DTypeName DT>
TensorStorage<DT>::TensorStorage(size_t size) : m_size(size) {
    /* TODO: use aligned allocation. */
    m_data_ptr = new cctype[size];
}

template <DTypeName DT>
TensorStorage<DT>::~TensorStorage() {
    if (m_data_ptr != nullptr) {
        delete []m_data_ptr;
        m_data_ptr = nullptr;
    }
}

template <DTypeName DT>
size_t TensorStorage<DT>::size() const {
    return m_size;
}

template <DTypeName DT>
size_t TensorStorage<DT>::memsize() const {
    return m_size * sizeof(cctype);
}

template <DTypeName DT>
const typename TensorStorage<DT>::cctype *TensorStorage<DT>::data_ptr() const {
    return m_data_ptr;
}

template <DTypeName DT>
typename TensorStorage<DT>::cctype *TensorStorage<DT>::mutable_data_ptr() {
    return m_data_ptr;
}

template <DTypeName DT>
std::ostream &operator << (std::ostream &out, const TensorStorage<DT> &storage) {
    out << "TensorStorage(dtype=" << DType<DT>::name << ", " << "size=" << storage.m_size << ", data_ptr=" << storage.m_data_ptr << ", values=[";
    for (ssize_t i = 0; i < std::min(TensorValueMaxPrint, storage.m_size); ++i) {
        out << (i == 0 ? "" : ", ") << storage.m_data_ptr[i];
    }
    if (storage.m_size > TensorValueMaxPrint) {
        out << ", ...";
    }
    out << "])";
    return out;
}

#define INSTANTIATE_FUNC(dtype) std::ostream &operator << <DTypeName::dtype> (std::ostream &out, const TensorStorage<DTypeName::dtype> &storage)
NCG_INSTANTIATE_DTYPE_ALL(INSTANTIATE_FUNC);
#undef INSTANTIATE_FUNC

NCG_INSTANTIATE_DTYPE_CLASS_ALL(TensorStorage);

Tensor::Tensor() : m_desc() {}
Tensor::Tensor(const TensorDesc &desc) : m_desc(desc) {}

TensorDesc &Tensor::desc() {
    return m_desc;
}

const TensorDesc &Tensor::desc() const {
    return m_desc;
}

template <DTypeName DT>
TensorImpl<DT> *Tensor::as() {
    return (dynamic_cast<TensorImpl<DT> *>(this));
}

#define INSTANTIATE_FUNC(dtype) TensorImpl<DTypeName::dtype> *Tensor::as()
NCG_INSTANTIATE_DTYPE_ALL(INSTANTIATE_FUNC);
#undef INSTANTIATE_FUNC

template <DTypeName DT>
const TensorImpl<DT> *Tensor::as() const {
    return (dynamic_cast<const TensorImpl<DT> *>(this));
}

#define INSTANTIATE_FUNC(dtype) const TensorImpl<DTypeName::dtype> *Tensor::as() const
NCG_INSTANTIATE_DTYPE_ALL(INSTANTIATE_FUNC);
#undef INSTANTIATE_FUNC

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

TensorPtr zeros(DTypeName dtype, const std::vector<size_t> &shape) {
    return fill(dtype, shape, 0);
}

TensorPtr ones(DTypeName dtype, const std::vector<size_t> &shape) {
    return fill(dtype, shape, 1);
}

} /* !namespace ncg */
