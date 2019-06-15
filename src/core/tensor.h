/*
 * tensor.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#pragma once

#include "core/common.h"
#include "core/datatype.h"
#include "core/tensor_desc.h"
#include "core/tensor_storage.h"

#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <vector>
#include <memory>
#include <limits>

namespace ncg {

template <DTypeName DT>
class TensorImpl;

class Tensor {
public:
    Tensor();
    Tensor(const TensorDesc &desc, std::shared_ptr<TensorStorage> storage, bool own_data=true, ssize_t data_ptr_offset=0);
    virtual ~Tensor() = default;

    TensorDesc &desc();
    const TensorDesc &desc(void) const;
    template <DTypeName DT>
    TensorImpl<DT> *as(void);
    template <DTypeName DT>
    const TensorImpl<DT> *as(void) const;

    std::shared_ptr<TensorStorage> storage();
    std::shared_ptr<const TensorStorage> storage() const;

    virtual void make_own_data() = 0;
    virtual void make_contiguous() = 0;

    friend std::ostream &operator << (std::ostream &out, const Tensor &tensor);

    template <typename ...Ints>
    inline const ssize_t index(Ints... args) const {
        auto indices = {args...};
        ncg_assert(indices.size() == m_desc.dim());
        ssize_t j = 0, i = 0;
        for (auto it = indices.begin(); it != indices.end(); ++it) j += (*it) * m_desc.stride(i++);
        return j;
    }
    inline const ssize_t elindex(ssize_t elindex) const {
        ssize_t ret = 0;
        for (ssize_t i = m_desc.dim() - 1; i >= 0; --i) {
            ret += (elindex % m_desc.shape(i)) * m_desc.stride(i);
            elindex /= m_desc.shape(i);
        }

        return ret;
    }

    bool own_data() const;
    ssize_t data_ptr_offset() const;

protected:
    TensorDesc m_desc;
    std::shared_ptr<TensorStorage> m_storage;
    bool m_own_data;
    ssize_t m_data_ptr_offset;
};

typedef std::shared_ptr<Tensor> TensorPtr;
typedef std::vector<std::shared_ptr<Tensor>> TensorVec;

template <DTypeName DT>
class TensorImpl : public Tensor {
public:
    using cctype = typename DType<DT>::cctype;

    TensorImpl() {}
    explicit TensorImpl(const TensorDesc &desc, std::shared_ptr<TensorStorage> storage, bool own_data=true, ssize_t data_ptr_offset=0) : Tensor(desc, storage, own_data, data_ptr_offset) {
        ncg_assert(storage->dtype() == desc.dtype());
    }
    explicit TensorImpl(const TensorDesc &desc, TensorStorage *storage, bool own_data=true, ssize_t data_ptr_offset=0) : Tensor(desc, nullptr, own_data, data_ptr_offset) {
        ncg_assert(storage->dtype() == desc.dtype());
        m_storage = std::shared_ptr<TensorStorage>(storage);
    }
    explicit TensorImpl(const TensorDesc &desc, cctype *data_ptr, bool own_data=true, ssize_t data_ptr_offset=0) : Tensor(desc, nullptr, own_data, data_ptr_offset) {
        auto storage = new TensorStorageImpl<DT>(data_ptr);
        m_storage = std::shared_ptr<TensorStorage>(storage);
    }
    virtual ~TensorImpl() = default;

    virtual void make_own_data() {
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

    virtual void make_contiguous() {
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

    template <typename ...Ints>
    inline const cctype &at(Ints... args) const {
        return data_ptr()[index(args...)];
    }
    template <typename ...Ints>
    inline cctype &mutable_at(Ints... args) {
        return mutable_data_ptr()[index(args...)];
    }

    inline const cctype &elat(ssize_t i) const {
        return data_ptr()[elindex(i)];
    }
    inline cctype &mutable_elat(ssize_t i) {
        return mutable_data_ptr()[elindex(i)];
    }

    inline const cctype *data_ptr() const {
        return storage_impl_()->data_ptr() + m_data_ptr_offset;
    }
    inline cctype *mutable_data_ptr() {
        make_own_data();
        return storage_impl_()->mutable_data_ptr() + m_data_ptr_offset;
    }

    inline friend std::ostream &operator << (std::ostream &out, const TensorImpl<DT> &tensor) {
        out << "Tensor(desc=" << tensor.m_desc << ", storage=" << *(tensor.storage()) << ", own_data=" << (tensor.m_own_data ? 'T' : 'F') << ", data_ptr_offset=" << tensor.m_data_ptr_offset << ")";
        return out;
    }

protected:
    const TensorStorageImpl<DT> *storage_impl_() const {
        return dynamic_cast<const TensorStorageImpl<DT> *>(m_storage.get());
    }

    TensorStorageImpl<DT> *storage_impl_() {
        return dynamic_cast<TensorStorageImpl<DT> *>(m_storage.get());
    }
};

TensorPtr tensor(const TensorDesc &desc, std::shared_ptr<TensorStorage> storage, bool own_data=true, ssize_t data_ptr_offset=0);
TensorPtr empty(DTypeName dtype, const ShapeVec &shape);

template <typename ValueT = double>
TensorPtr fill(DTypeName dtype, const ShapeVec &shape, ValueT value) {
    auto s = empty(dtype, shape);

#define FILL_DTYPE_CASE(dtype_name) do { \
    auto s_dtype = s->template as<DTypeName::dtype_name>();\
    for (ssize_t i = 0; i < s_dtype->desc().numel(); ++i) s_dtype->mutable_elat(i) = value; \
} while(0)
NCG_DTYPE_SWITCH_ALL(dtype, FILL_DTYPE_CASE);
#undef FILL_DTYPE_CASE

    return s;
}

TensorPtr zeros(DTypeName dtype, const ShapeVec &shape);
TensorPtr ones(DTypeName dtype, const ShapeVec &shape);

template <typename ValueT = double>
TensorPtr scalar(DTypeName dtype, ValueT value = 0) {
    auto s = empty(dtype, {});

#define SCALAR_DTYPE_CASE(dtype_name) s->template as<DTypeName::dtype_name>()->mutable_data_ptr()[0] = value
NCG_DTYPE_SWITCH_ALL(dtype, SCALAR_DTYPE_CASE);
#undef SCALAR_DTYPE_CASE

    return s;
}

template <typename ValueT = double>
TensorPtr fromcc(DTypeName dtype, ValueT value = 0) {
    return scalar(dtype, value);
}

template <typename ValueT = double>
TensorPtr fromcc(DTypeName dtype, std::vector<ValueT> values = 0) {
    auto s = empty(dtype, {static_cast<ssize_t>(values.size())});

#define FROMCC_DTYPE_CASE(dtype_name) do { \
    auto data_ptr = s->template as<DTypeName::dtype_name>()->mutable_data_ptr(); \
    for (ssize_t i = 0; i < values.size(); ++i) { data_ptr[i] = values[i]; } \
} while(0)
NCG_DTYPE_SWITCH_ALL(dtype, FROMCC_DTYPE_CASE);
#undef FROMCC_DTYPE_CASE

    return s;
}

template <typename ValueT = double>
TensorPtr fromcc(DTypeName dtype, std::vector<std::vector<ValueT>> values = 0) {
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

TensorPtr arange(DTypeName dtype, int64_t begin, int64_t end = std::numeric_limits<int64_t>::min(), int64_t step=1);

} /* !namespace ncg */

