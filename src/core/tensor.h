/*
 * tensor.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#ifndef CORE_TENSOR_H
#define CORE_TENSOR_H

#include "core/common.h"
#include "core/datatype.h"

#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <vector>
#include <memory>

namespace ncg {

const size_t TensorMaxDim = 15;
const size_t TensorShape0 = static_cast<size_t>(-1);
const size_t TensorValueMaxPrint = 8;

class TensorDesc {
public:
    TensorDesc();
    TensorDesc(DTypeName dtype, const std::vector<size_t> &shape, const std::vector<size_t> &stride = {});
    virtual ~TensorDesc() = default;

    DTypeName dtype() const;

    size_t dim() const;

    std::vector<size_t> shape_vec() const;
    size_t *shape();
    const size_t *shape() const;
    size_t *stride();
    const size_t *stride() const;

    size_t &shape(ssize_t i);
    size_t shape(ssize_t i) const;
    size_t &stride(ssize_t i);
    size_t stride(ssize_t i) const;

    bool is_continugous();
    size_t numel() const;
    bool is_compatible(const TensorDesc &rhs);
    friend std::ostream &operator << (std::ostream &out, const TensorDesc &desc);

protected:
    DTypeName m_dtype;
    size_t m_shape[TensorMaxDim + 1];
    size_t m_stride[TensorMaxDim + 1];
};

template <DTypeName DT> class TensorStorage;
template <DTypeName DT> std::ostream &operator << (std::ostream &out, const TensorStorage<DT> &storage);

template <DTypeName DT>
class TensorStorage {
public:
    using cctype = typename DType<DT>::cctype;

    TensorStorage();
    explicit TensorStorage(cctype *data_ptr, size_t size);
    explicit TensorStorage(size_t size);

    /* NB: delete the copy-constructor and move-constructor */
    TensorStorage(const TensorStorage<DT> &) = delete;
    TensorStorage(TensorStorage<DT> &&) = delete;

    virtual ~TensorStorage();

    size_t size() const;
    size_t memsize() const;
    const cctype *data_ptr() const;
    cctype *mutable_data_ptr();

    friend std::ostream &operator << <> (std::ostream &out, const TensorStorage<DT> &storage);

protected:
    cctype *m_data_ptr;
    size_t m_size;
};

template <DTypeName DT>
class TensorImpl;

class Tensor {
public:
    Tensor();
    Tensor(const TensorDesc &desc);
    virtual ~Tensor() = default;

    TensorDesc &desc();
    const TensorDesc &desc(void) const;
    template <DTypeName DT>
    TensorImpl<DT> *as(void);
    template <DTypeName DT>
    const TensorImpl<DT> *as(void) const;

protected:
    TensorDesc m_desc;
};

typedef std::shared_ptr<Tensor> TensorPtr;
typedef std::vector<std::shared_ptr<Tensor>> TensorVec;

template <DTypeName DT>
class TensorImpl : public Tensor {
public:
    using cctype = typename DType<DT>::cctype;

    TensorImpl() : m_storage() {}
    explicit TensorImpl(const TensorDesc &desc, std::shared_ptr<TensorStorage<DT>> storage) : Tensor(desc), m_storage(storage) {}
    explicit TensorImpl(const TensorDesc &desc, TensorStorage<DT> *storage) : Tensor(desc), m_storage(storage) {}
    explicit TensorImpl(const TensorDesc &desc, cctype *data_ptr) : Tensor(desc) {
        m_storage = std::make_shared<TensorStorage<DT>>(data_ptr);
    }
    virtual ~TensorImpl() = default;

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
        for (ssize_t i = 0; i < m_desc.dim(); ++i) {
            ret += elindex / m_desc.shape(i) * m_desc.stride(i);
            elindex %= m_desc.shape(i);
        }
        return ret;
    }

    template <typename ...Ints>
    inline const cctype &at(Ints... args) const {
        return data_ptr()[index(args...)];
    }
    template <typename ...Ints>
    inline cctype &at(Ints... args) {
        return mutable_data_ptr()[index(args...)];
    }

    inline const cctype &elat(ssize_t i) const {
        return data_ptr()[elindex(i)];
    }
    inline cctype &elat(ssize_t i) {
        return mutable_data_ptr()[elindex(i)];
    }

    inline const TensorStorage<DT> &storage() const { return *m_storage; }
    inline TensorStorage<DT> &storage() { return *m_storage; }

    inline const cctype *data_ptr() const {
        return m_storage->data_ptr();
    }
    inline cctype *mutable_data_ptr() {
        return m_storage->mutable_data_ptr();
    }

    inline friend std::ostream &operator << (std::ostream &out, const TensorImpl<DT> &tensor) {
        out << "Tensor(desc=" << tensor.m_desc << ", storage=" << tensor.storage()  << ")";
        return out;
    }

protected:
    std::shared_ptr<TensorStorage<DT>> m_storage;
};

template <DTypeName DT>
TensorPtr tensor(const TensorDesc &desc, std::shared_ptr<TensorStorage<DT>> storage) {
    ncg_assert(desc.dtype() == DT);
    return TensorPtr(new TensorImpl<DT>(desc, storage));
}

TensorPtr empty(DTypeName dtype, const std::vector<size_t> &shape);

template <typename ValueT = double>
TensorPtr fill(DTypeName dtype, const std::vector<size_t> &shape, ValueT value) {
    auto s = empty(dtype, shape);

#define FILL_DTYPE_CASE(dtype_name) do { \
    auto s_dtype = s->as<DTypeName::dtype_name>();\
    for (ssize_t i = 0; i < s_dtype->desc().numel(); ++i) s_dtype->elat(i) = value; \
} while(0)
NCG_SWITCH_DTYPE_ALL(dtype, FILL_DTYPE_CASE);
#undef FILL_DTYPE_CASE

    return s;
}

TensorPtr zeros(DTypeName dtype, const std::vector<size_t> &shape);
TensorPtr ones(DTypeName dtype, const std::vector<size_t> &shape);

template <typename ValueT = double>
TensorPtr scalar(DTypeName dtype, ValueT value = 0) {
    auto s = empty(dtype, {});

#define SCALAR_DTYPE_CASE(dtype_name) s->as<DTypeName::dtype_name>()->mutable_data_ptr()[0] = value
NCG_SWITCH_DTYPE_ALL(dtype, SCALAR_DTYPE_CASE);
#undef SCALAR_DTYPE_CASE

    return s;
}

} /* !namespace ncg */

#endif /* !CORE_TENSOR_H */
