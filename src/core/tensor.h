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
#include <limits>

namespace ncg {

const size_t TensorMaxDim = 15;
const size_t TensorValueMaxPrint = 8;

const ssize_t TensorShape0 = std::numeric_limits<ssize_t>::min();
const ssize_t NewAxis = std::numeric_limits<ssize_t>::max();

class shape_vec : public std::vector<ssize_t> {
public:
    using std::vector<ssize_t>::vector;
    friend std::ostream &operator << (std::ostream &out, const shape_vec &shape);
};

class TensorDesc {
public:
    TensorDesc();
    TensorDesc(DTypeName dtype, const class shape_vec &shape, const class shape_vec &stride = {});
    virtual ~TensorDesc() = default;

    DTypeName dtype() const;

    size_t dim() const;

    class shape_vec shape_vec() const;
    ssize_t *shape();
    const ssize_t *shape() const;
    ssize_t *stride();
    const ssize_t *stride() const;

    ssize_t &shape(ssize_t i);
    ssize_t shape(ssize_t i) const;
    ssize_t &stride(ssize_t i);
    ssize_t stride(ssize_t i) const;

    void set_default_stride();

    bool is_continugous();
    size_t numel() const;
    bool is_compatible(const TensorDesc &rhs, bool allow_broadcast=false);
    friend std::ostream &operator << (std::ostream &out, const TensorDesc &desc);

protected:
    DTypeName m_dtype;
    ssize_t m_shape[TensorMaxDim + 1];
    ssize_t m_stride[TensorMaxDim + 1];
};

template <DTypeName DT> class TensorStorage;
template <DTypeName DT> std::ostream &operator << (std::ostream &out, const TensorStorage<DT> &storage);

class TensorStorage {
public:
    TensorStorage() = default;
    virtual ~TensorStorage() = default;

    virtual TensorStorage *clone() const;
    virtual size_t size() const;
    virtual size_t memsize() const;
};

template <DTypeName DT>
class TensorStorageImpl : public TensorStorage {
public:
    using cctype = typename DType<DT>::cctype;

    TensorStorageImpl();
    explicit TensorStorageImpl(cctype *data_ptr, size_t size);
    explicit TensorStorageImpl(size_t size);

    /* NB: delete the copy-constructor and move-constructor */
    TensorStorageImpl(const TensorStorage<DT> &) = delete;
    TensorStorageImpl(TensorStorage<DT> &&) = delete;

    virtual ~TensorStorageImpl();

    virtual TensorStorage *clone() const;
    virtual size_t size() const;
    virtual size_t memsize() const;

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

    virtual void make_own_data();
    virtual void make_contiguous();

    virtual const TensorStorage *storage() const;
    virtual TensorStorage *storage();

    inline friend std::ostream &operator << (std::ostream &out, const TensorImpl<DT> &tensor) {
        out << "Tensor(desc=" << tensor.m_desc << ", storage=" << *(tensor.storage())  << ")";
        return out;
    }

protected:
    TensorDesc m_desc;
};

typedef std::shared_ptr<Tensor> TensorPtr;
typedef std::vector<std::shared_ptr<Tensor>> TensorVec;

template <DTypeName DT>
class TensorImpl : public Tensor {
public:
    using cctype = typename DType<DT>::cctype;

    TensorImpl() : m_storage(), m_data_ptr_offset(0), m_own_data(true) {}
    explicit TensorImpl(const TensorDesc &desc, std::shared_ptr<TensorStorageImpl<DT>> storage, bool own_data=true, ssize_t data_ptr_offset=0) : Tensor(desc), m_storage(storage), m_own_data(own_data), m_data_ptr_offset(data_ptr_offset) {}
    explicit TensorImpl(const TensorDesc &desc, TensorStorageImpl<DT> *storage, bool own_data=true, ssize_t data_ptr_offset=0) : Tensor(desc), m_storage(storage), m_own_data(own_data), m_data_ptr_offset(data_ptr_offset) {}
    explicit TensorImpl(const TensorDesc &desc, cctype *data_ptr, bool own_data=true, ssize_t data_ptr_offset=0) : Tensor(desc), m_own_data(own_data), m_data_ptr_offset(data_ptr_offset) {
        m_storage = std::make_shared<TensorStorageImpl<DT>>(data_ptr);
    }
    virtual ~TensorImpl() = default;

    virtual void make_own_data() {
        if (!m_own_data) {
            if (m_desc.is_continugous()) {
                m_storage = std::shared_ptr<TensorStorageImpl<DT>>(m_storage->clone());
                m_own_data = true;
            } else {
                make_contiguous();
            }
        }
    }

    virtual void make_contiguous() {
        if (m_desc.is_continugous()) {
            return ;
        } else {
            auto storage = std::make_shared<TensorStorageImpl<DT>>(m_storage->size());
            for (ssize_t i = 0; i < m_desc.numel(); ++i) {
                storage->mutable_data_ptr()[i] = elat(i);
            }

            m_desc.set_default_stride();
            m_storage = storage;
            m_own_data = true;
        }
    }

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

    inline const cctype *data_ptr() const {
        return m_storage->data_ptr() + m_data_ptr_offset;
    }
    inline cctype *mutable_data_ptr() {
        make_own_data();
        return m_storage->mutable_data_ptr() + m_data_ptr_offset;
    }

    virtual const TensorStorage *storage() const { return *m_storage; }
    virtual TensorStorage *storage() { return *m_storage; }

protected:
    ssize_t m_data_ptr_offset;
    bool m_own_data;
    std::shared_ptr<TensorStorage<DT>> m_storage;
};

template <DTypeName DT>
TensorPtr tensor(const TensorDesc &desc, std::shared_ptr<TensorStorage<DT>> storage, bool own_data=true, ssize_t data_ptr_offset=0) {
    ncg_assert(desc.dtype() == DT);
    return TensorPtr(new TensorImpl<DT>(desc, storage, own_data, data_ptr_offset));
}

TensorPtr empty(DTypeName dtype, const shape_vec &shape);

template <typename ValueT = double>
TensorPtr fill(DTypeName dtype, const shape_vec &shape, ValueT value) {
    auto s = empty(dtype, shape);

#define FILL_DTYPE_CASE(dtype_name) do { \
    auto s_dtype = s->as<DTypeName::dtype_name>();\
    for (ssize_t i = 0; i < s_dtype->desc().numel(); ++i) s_dtype->elat(i) = value; \
} while(0)
NCG_SWITCH_DTYPE_ALL(dtype, FILL_DTYPE_CASE);
#undef FILL_DTYPE_CASE

    return s;
}

TensorPtr zeros(DTypeName dtype, const shape_vec &shape);
TensorPtr ones(DTypeName dtype, const shape_vec &shape);

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
