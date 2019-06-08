/*
 * tensor.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include "tensor.h"
#include <cstring>

namespace ncg {

std::ostream &operator << (std::ostream &out, const shape_vec &shape) {
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

TensorDesc::TensorDesc(DTypeName dtype, const class shape_vec &shape, const class shape_vec &stride) : m_dtype(dtype) {
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

size_t TensorDesc::dim(void) const {
    size_t i;
    for (i = 0; i <= TensorMaxDim; ++i)
        if (m_shape[i] == TensorShape0) break;
    return i;
}

class shape_vec TensorDesc::shape_vec(void) const {
    return ncg::shape_vec(m_shape, m_shape + dim());
}

ssize_t *TensorDesc::shape(void) {
    return m_shape;
}

const ssize_t *TensorDesc::shape(void) const {
    return m_shape;
}

class shape_vec TensorDesc::stride_vec(void) const {
    return ncg::shape_vec(m_stride, m_stride + dim());
}

ssize_t *TensorDesc::stride(void) {
    return m_stride;
}

const ssize_t *TensorDesc::stride(void) const {
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

class shape_vec TensorDesc::get_default_stride() const {
    size_t d = dim();
    ncg::shape_vec stride_vec(d);

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
        if (m_shape[i] == TensorShape0) break;
        n *= m_shape[i];
    }
    return n;
}

bool TensorDesc::is_compatible(const TensorDesc &rhs, bool allow_broadcast) {
    if (m_dtype != rhs.m_dtype) return false;
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

TensorStorage::TensorStorage(DTypeName dtype) : m_dtype(dtype) {

}

DTypeName TensorStorage::dtype() const {
    return m_dtype;
}

std::ostream &operator << (std::ostream &out, const TensorStorage &storage) {
    #define COUT_STORAGE_DTYPE_CASE(dtype) out << dynamic_cast<const TensorStorageImpl<DTypeName::dtype> &>(storage);
    NCG_SWITCH_DTYPE_ALL(storage.dtype(), COUT_STORAGE_DTYPE_CASE)
    #undef COUT_STORAGE_DTYPE_CASE
    return out;
}

template <DTypeName DT>
TensorStorageImpl<DT>::TensorStorageImpl() : TensorStorage(DT) {

}

template <DTypeName DT>
TensorStorageImpl<DT>::TensorStorageImpl(cctype *data_ptr, size_t size) : TensorStorage(DT), m_data_ptr(data_ptr), m_size(size) {

}

template <DTypeName DT>
TensorStorageImpl<DT>::TensorStorageImpl(size_t size) : TensorStorage(DT), m_size(size) {
    /* TODO: use aligned allocation. */
    m_data_ptr = new cctype[size];
}

template <DTypeName DT>
TensorStorageImpl<DT>::~TensorStorageImpl() {
    if (m_data_ptr != nullptr) {
        delete []m_data_ptr;
        m_data_ptr = nullptr;
    }
}

template <DTypeName DT>
TensorStorage *TensorStorageImpl<DT>::clone(ssize_t start, ssize_t length) const {
    auto ret = new TensorStorageImpl<DT>();

    if (length > m_size - start) {
        length = m_size - start;
    }

    if (m_data_ptr != nullptr) {
        ret->m_data_ptr = new cctype[length];
        ret->m_size = length;

        memcpy(ret->m_data_ptr, m_data_ptr + start, m_size * sizeof(cctype));
    }
    return static_cast<TensorStorage *>(ret);
}

template <DTypeName DT>
size_t TensorStorageImpl<DT>::size() const {
    return m_size;
}

template <DTypeName DT>
size_t TensorStorageImpl<DT>::memsize() const {
    return m_size * sizeof(cctype);
}

template <DTypeName DT>
const typename TensorStorageImpl<DT>::cctype *TensorStorageImpl<DT>::data_ptr() const {
    return m_data_ptr;
}

template <DTypeName DT>
typename TensorStorageImpl<DT>::cctype *TensorStorageImpl<DT>::mutable_data_ptr() {
    return m_data_ptr;
}

template <DTypeName DT>
std::ostream &operator << (std::ostream &out, const TensorStorageImpl<DT> &storage) {
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

#define INSTANTIATE_FUNC(dtype) std::ostream &operator << <DTypeName::dtype> (std::ostream &out, const TensorStorageImpl<DTypeName::dtype> &storage)
NCG_INSTANTIATE_DTYPE_ALL(INSTANTIATE_FUNC);
#undef INSTANTIATE_FUNC

NCG_INSTANTIATE_DTYPE_CLASS_ALL(TensorStorageImpl);

Tensor::Tensor() : m_desc(), m_storage(), m_own_data(false), m_data_ptr_offset(0) {}
Tensor::Tensor(const TensorDesc &desc, std::shared_ptr<TensorStorage> storage, bool own_data, ssize_t data_ptr_offset) : m_desc(desc), m_storage(storage), m_own_data(own_data), m_data_ptr_offset(data_ptr_offset) {}

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

std::shared_ptr<TensorStorage> Tensor::storage() {
    return m_storage;
}

std::shared_ptr<const TensorStorage> Tensor::storage() const {
    return m_storage;
}

bool Tensor::own_data() const {
    return m_own_data;
}

ssize_t Tensor::data_ptr_offset() const {
    return  m_data_ptr_offset;
}


std::ostream &operator << (std::ostream &out, const Tensor &tensor) {
#define COUT_TENSOR_DTYPE_CASE(dtype) out << dynamic_cast<const TensorImpl<DTypeName::dtype> &>(tensor);
NCG_SWITCH_DTYPE_ALL(tensor.desc().dtype(), COUT_TENSOR_DTYPE_CASE)
#undef COUT_TENSOR_DTYPE_CASE
    return out;
}

TensorPtr tensor(const TensorDesc &desc, std::shared_ptr<TensorStorage> storage, bool own_data, ssize_t data_ptr_offset) {
    ncg_assert(desc.dtype() == storage->dtype());
    Tensor *tensor = nullptr;

#define TENSOR_DTYPE_CASE(dtype) tensor = static_cast<Tensor *>(new TensorImpl<DTypeName::dtype>(desc, storage, own_data, data_ptr_offset));
NCG_SWITCH_DTYPE_ALL(desc.dtype(), TENSOR_DTYPE_CASE)
#undef TENSOR_DTYPE_CASE

    return TensorPtr(tensor);
}

TensorPtr empty(DTypeName dtype, const shape_vec &shape) {
    TensorDesc desc(dtype, shape);

#define EMPTY_DTYPE_CASE(dtype) return std::shared_ptr<Tensor>( \
        static_cast<Tensor *>(new TensorImpl<DTypeName::dtype>(\
            desc, new TensorStorageImpl<DTypeName::dtype>(desc.numel()) \
        )) \
    )
NCG_SWITCH_DTYPE_ALL(dtype, EMPTY_DTYPE_CASE)
#undef EMPTY_DTYPE_CASE

    return std::shared_ptr<Tensor>(nullptr);
}

TensorPtr zeros(DTypeName dtype, const shape_vec &shape) {
    return fill(dtype, shape, 0);
}

TensorPtr ones(DTypeName dtype, const shape_vec &shape) {
    return fill(dtype, shape, 1);
}

TensorPtr arange(DTypeName dtype, int64_t begin, int64_t end, int64_t step) {
    if (end == std::numeric_limits<int64_t>::min()) {
        end = begin;
        begin = 0;
    }

    auto s = empty(dtype, {(end - begin - 1) / step + 1});

#define ARANGE_DTYPE_CASE(dtype_name) do { \
    auto s_dtype = s->template as<DTypeName::dtype_name>();\
    for (ssize_t i = 0; i < (end - begin - 1) / step + 1; ++i) { s_dtype->mutable_elat(i) = static_cast<DType<DTypeName::dtype_name>::cctype>(begin + step * i); } \
} while(0)
NCG_SWITCH_DTYPE_ALL(dtype, ARANGE_DTYPE_CASE);
#undef ARANGE_DTYPE_CASE

    return s;
}

} /* !namespace ncg */
