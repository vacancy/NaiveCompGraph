/*
 * tensor_storage.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include "core/datatype.h"
#include "core/tensor_storage.h"

namespace ncg {

TensorStorage::TensorStorage(DTypeName dtype) : m_dtype(dtype) {

}

DTypeName TensorStorage::dtype() const {
    return m_dtype;
}

std::ostream &operator << (std::ostream &out, const TensorStorage &storage) {
#define COUT_STORAGE_DTYPE_CASE(dtype_name) out << dynamic_cast<const TensorStorageImpl<DTypeName::dtype_name> &>(storage);
NCG_DTYPE_SWITCH_ALL(storage.dtype(), COUT_STORAGE_DTYPE_CASE)
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
    out << "TensorStorage(dtype=" << get_dtype_name(DT) << ", " << "size=" << storage.m_size << ", data_ptr=" << storage.m_data_ptr << ", values=[";
    for (ssize_t i = 0; i < std::min(TensorValueMaxPrint, storage.m_size); ++i) {
        out << (i == 0 ? "" : ", ") << storage.m_data_ptr[i];
    }
    if (storage.m_size > TensorValueMaxPrint) {
        out << ", ...";
    }
    out << "])";
    return out;
}

#define INSTANTIATE_FUNC(dtype_name) std::ostream &operator << <DTypeName::dtype_name> (std::ostream &out, const TensorStorageImpl<DTypeName::dtype_name> &storage)
NCG_DTYPE_INSTANTIATE_ALL(INSTANTIATE_FUNC);
#undef INSTANTIATE_FUNC

NCG_DTYPE_INSTANTIATE_CLASS_ALL(TensorStorageImpl);

} /* !namespace ncg */
