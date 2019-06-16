/*
 * tensor_impl.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#pragma once

#include "core/tensor.h"

namespace ncg {

template <DTypeName DT>
TensorImpl<DT> *Tensor::as() {
    return (dynamic_cast<TensorImpl<DT> *>(this));
}

#define INSTANTIATE_FUNC(dtype_name) TensorImpl<DTypeName::dtype_name> *Tensor::as()
NCG_DTYPE_INSTANTIATE_ALL(INSTANTIATE_FUNC);
#undef INSTANTIATE_FUNC

template <DTypeName DT>
const TensorImpl<DT> *Tensor::as() const {
    return (dynamic_cast<const TensorImpl<DT> *>(this));
}

#define INSTANTIATE_FUNC(dtype_name) const TensorImpl<DTypeName::dtype_name> *Tensor::as() const
NCG_DTYPE_INSTANTIATE_ALL(INSTANTIATE_FUNC);
#undef INSTANTIATE_FUNC

template <typename ...Ints>
const ssize_t Tensor::index(Ints... args) const {
    auto indices = {args...};
    ncg_assert(indices.size() == m_desc.dim());
    ssize_t j = 0, i = 0;
    for (auto it = indices.begin(); it != indices.end(); ++it) j += (*it) * m_desc.stride(i++);
    return j;
}

template <DTypeName DT>
TensorImpl<DT>::TensorImpl() {}

template <DTypeName DT>
TensorImpl<DT>::TensorImpl(const TensorDesc &desc, std::shared_ptr<TensorStorage> storage, bool own_data, ssize_t data_ptr_offset) : Tensor(desc, storage, own_data, data_ptr_offset) {
    ncg_assert(storage->dtype() == desc.dtype());
}

template <DTypeName DT>
TensorImpl<DT>::TensorImpl(const TensorDesc &desc, TensorStorage *storage, bool own_data, ssize_t data_ptr_offset) : Tensor(desc, nullptr, own_data, data_ptr_offset) {
    ncg_assert(storage->dtype() == desc.dtype());
    m_storage = std::shared_ptr<TensorStorage>(storage);
}

template <DTypeName DT>
TensorImpl<DT>::TensorImpl(const TensorDesc &desc, typename TensorImpl<DT>::cctype *data_ptr, bool own_data, ssize_t data_ptr_offset) : Tensor(desc, nullptr, own_data, data_ptr_offset) {
    auto storage = new TensorStorageImpl<DT>(data_ptr);
    m_storage = std::shared_ptr<TensorStorage>(storage);
}

template <DTypeName DT>
void TensorImpl<DT>::make_own_data() {
    if (!m_own_data) {
        if (m_desc.is_continugous()) {
            m_storage = std::shared_ptr<TensorStorage>(m_storage->clone(m_data_ptr_offset, m_desc.numel()));
            m_own_data = true;
            m_data_ptr_offset = 0;
        } else {
            make_contiguous();
        }
    }
}

template <DTypeName DT>
void TensorImpl<DT>::make_contiguous() {
    if (m_desc.is_continugous()) {
        return ;
    } else {
        auto storage = new TensorStorageImpl<DT>(m_desc.numel());
        for (ssize_t i = 0; i < m_desc.numel(); ++i) {
            storage->mutable_data_ptr()[i] = elat(i);
        }

        m_desc.set_default_stride();
        m_storage = std::shared_ptr<TensorStorage>(static_cast<TensorStorage *>(storage));
        m_own_data = true;
        m_data_ptr_offset = 0;
    }
}

template <DTypeName DT>
template <typename ...Ints>
const typename TensorImpl<DT>::cctype &TensorImpl<DT>::at(Ints... args) const {
    return data_ptr()[index(args...)];
}

template <DTypeName DT>
template <typename ...Ints>
typename TensorImpl<DT>::cctype &TensorImpl<DT>::mutable_at(Ints... args) {
    return mutable_data_ptr()[index(args...)];
}

template <DTypeName DT>
const typename TensorImpl<DT>::cctype &TensorImpl<DT>::elat(ssize_t i) const {
    return data_ptr()[elindex(i)];
}

template <DTypeName DT>
typename TensorImpl<DT>::cctype &TensorImpl<DT>::mutable_elat(ssize_t i) {
    return mutable_data_ptr()[elindex(i)];
}

template <DTypeName DT>
const typename TensorImpl<DT>::cctype *TensorImpl<DT>::data_ptr() const {
    return storage_impl_()->data_ptr() + m_data_ptr_offset;
}

template <DTypeName DT>
typename TensorImpl<DT>::cctype *TensorImpl<DT>::mutable_data_ptr() {
    make_own_data();
    return storage_impl_()->mutable_data_ptr() + m_data_ptr_offset;
}

template <DTypeName DT>
const TensorStorageImpl<DT> *TensorImpl<DT>::storage_impl_() const {
    return dynamic_cast<const TensorStorageImpl<DT> *>(m_storage.get());
}

template <DTypeName DT>
TensorStorageImpl<DT> *TensorImpl<DT>::storage_impl_() {
    return dynamic_cast<TensorStorageImpl<DT> *>(m_storage.get());
}

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, TensorPtr>::type
as_tensor(T value) {
    return TensorPtr(fromcc(::ncg::CCType<T>::identifier, value));
}
template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, TensorPtr>::type
as_tensor(std::vector<T> value) {
    return TensorPtr(fromcc(::ncg::CCType<T>::identifier, value));
}
template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, TensorPtr>::type
as_tensor(std::vector<std::vector<T>> value) {
    return TensorPtr(fromcc(::ncg::CCType<T>::identifier, value));
}

template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type*>
TensorPtr::TensorPtr(T value) : TensorPtr(as_tensor(value)) {}
template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type*>
TensorPtr::TensorPtr(std::vector<T> value) : TensorPtr(as_tensor(value)) {}
template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type*>
TensorPtr::TensorPtr(std::vector<std::vector<T>> value) : TensorPtr(as_tensor(value)) {}

template <typename ValueT>
typename std::enable_if<std::is_arithmetic<ValueT>::value, TensorPtr>::type
fill(DTypeName dtype, const ShapeVec &shape, ValueT value) {
    auto s = empty(dtype, shape);

#define FILL_DTYPE_CASE(dtype_name) do { \
    auto s_dtype = s->template as<DTypeName::dtype_name>();\
    for (ssize_t i = 0; i < s_dtype->desc().numel(); ++i) s_dtype->mutable_elat(i) = value; \
} while(0)
NCG_DTYPE_SWITCH_ALL(dtype, FILL_DTYPE_CASE);
#undef FILL_DTYPE_CASE

    return s;
}

template <typename ValueT>
typename std::enable_if<std::is_arithmetic<ValueT>::value, TensorPtr>::type
scalar(DTypeName dtype, ValueT value) {
    auto s = empty(dtype, {});

#define SCALAR_DTYPE_CASE(dtype_name) s->template as<DTypeName::dtype_name>()->mutable_data_ptr()[0] = value
NCG_DTYPE_SWITCH_ALL(dtype, SCALAR_DTYPE_CASE);
#undef SCALAR_DTYPE_CASE

    return s;
}

template <typename ValueT>
typename std::enable_if<std::is_arithmetic<ValueT>::value, TensorPtr>::type
fromcc(DTypeName dtype, ValueT value) {
    return scalar(dtype, value);
}

template <typename ValueT>
typename std::enable_if<std::is_arithmetic<ValueT>::value, TensorPtr>::type
fromcc(DTypeName dtype, std::vector<ValueT> values) {
    auto s = empty(dtype, {static_cast<ssize_t>(values.size())});

#define FROMCC_DTYPE_CASE(dtype_name) do { \
    auto data_ptr = s->template as<DTypeName::dtype_name>()->mutable_data_ptr(); \
    for (ssize_t i = 0; i < values.size(); ++i) { data_ptr[i] = values[i]; } \
} while(0)
NCG_DTYPE_SWITCH_ALL(dtype, FROMCC_DTYPE_CASE);
#undef FROMCC_DTYPE_CASE

    return s;
}

template <typename ValueT>
typename std::enable_if<std::is_arithmetic<ValueT>::value, TensorPtr>::type
fromcc(DTypeName dtype, std::vector<std::vector<ValueT>> values) {
    ncg_assert(values.size() > 0);
    for (ssize_t i = 0; i < values.size(); ++i) {
        ncg_assert(values[i].size() == values[0].size());
    }
    auto s = empty(dtype, {static_cast<ssize_t>(values.size()), static_cast<ssize_t>(values[0].size())});

#define FROMCC_DTYPE_CASE(dtype_name) do { \
    auto data_ptr = s->template as<DTypeName::dtype_name>()->mutable_data_ptr(); \
    for (ssize_t i = 0, k = 0; i < values.size(); ++i) { \
        for (ssize_t j = 0; j < values[i].size(); ++j) data_ptr[k++] = values[i][j]; \
    } \
} while(0)
NCG_DTYPE_SWITCH_ALL(dtype, FROMCC_DTYPE_CASE);
#undef FROMCC_DTYPE_CASE

    return s;
}

template <typename ValueT=double>
ValueT tocc_scalar(TensorPtr tensor) {
    ncg_assert(tensor->desc().dim() == 0);

#define TOCC_DTYPE_CASE(dtype_name) do { \
    auto data_ptr = tensor->template as<DTypeName::dtype_name>()->data_ptr(); \
    return data_ptr[0]; \
} while(0)
NCG_DTYPE_SWITCH_ALL(tensor->desc().dtype(), TOCC_DTYPE_CASE);
#undef TOCC_DTYPE_CASE
}

template <typename ValueT=double>
std::vector<ValueT> tocc_vector(TensorPtr tensor) {
    ncg_assert(tensor->desc().dim() == 1);
    std::vector<ValueT> output(tensor->desc().shape(0));

#define TOCC_DTYPE_CASE(dtype_name) do { \
    auto tensor_dtype = tensor->template as<DTypeName::dtype_name>(); \
    for (ssize_t i = 0; i < output.size(); ++i) output[i] = tensor_dtype->at(i); \
} while(0)
NCG_DTYPE_SWITCH_ALL(tensor->desc().dtype(), TOCC_DTYPE_CASE);
#undef TOCC_DTYPE_CASE

    return output;
}

} /* !namespace ncg */

