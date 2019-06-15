/*
 * tensor.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include "core/tensor.h"

#include <cstring>

namespace ncg {

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
#define COUT_TENSOR_DTYPE_CASE(dtype_name) out << dynamic_cast<const TensorImpl<DTypeName::dtype_name> &>(tensor);
NCG_DTYPE_SWITCH_ALL(tensor.desc().dtype(), COUT_TENSOR_DTYPE_CASE)
#undef COUT_TENSOR_DTYPE_CASE
    return out;
}

TensorPtr tensor(const TensorDesc &desc, std::shared_ptr<TensorStorage> storage, bool own_data, ssize_t data_ptr_offset) {
    ncg_assert(desc.dtype() == storage->dtype());
    Tensor *tensor = nullptr;

#define TENSOR_DTYPE_CASE(dtype_name) tensor = static_cast<Tensor *>(new TensorImpl<DTypeName::dtype_name>(desc, storage, own_data, data_ptr_offset));
NCG_DTYPE_SWITCH_ALL(desc.dtype(), TENSOR_DTYPE_CASE)
#undef TENSOR_DTYPE_CASE

    return TensorPtr(tensor);
}

TensorPtr empty(DTypeName dtype, const ShapeVec &shape) {
    TensorDesc desc(dtype, shape);

#define EMPTY_DTYPE_CASE(dtype_name) return std::shared_ptr<Tensor>( \
        static_cast<Tensor *>(new TensorImpl<DTypeName::dtype_name>(\
            desc, new TensorStorageImpl<DTypeName::dtype_name>(desc.numel()) \
        )) \
    )
NCG_DTYPE_SWITCH_ALL(dtype, EMPTY_DTYPE_CASE)
#undef EMPTY_DTYPE_CASE

    return std::shared_ptr<Tensor>(nullptr);
}

TensorPtr zeros(DTypeName dtype, const ShapeVec &shape) {
    return fill(dtype, shape, 0);
}

TensorPtr ones(DTypeName dtype, const ShapeVec &shape) {
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
NCG_DTYPE_SWITCH_ALL(dtype, ARANGE_DTYPE_CASE);
#undef ARANGE_DTYPE_CASE

    return s;
}

} /* !namespace ncg */
