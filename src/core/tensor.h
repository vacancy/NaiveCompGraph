/*
 * tensor.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#ifndef TENSOR_H
#define TENSOR_H

#include "core/common.h"
#include "core/datatype.h"
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <memory>

namespace ncg {

const size_t TensorMaxDim = 15;
const size_t TensorShape0 = static_cast<size_t>(-1);

class TensorDesc {
public:
    TensorDesc() {
        m_dtype = DTypeName::UInt8;
        memset(m_shape, 0, sizeof(m_shape));
        memset(m_stride, 0, sizeof(m_stride));
    }

    TensorDesc(DTypeName dtype, const std::initializer_list<size_t> &shape, const std::initializer_list<size_t> &stride = {}) : m_dtype(dtype) {
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
                for (int i = d - 2; i >= 0; --i) {
                    m_stride[i] = m_stride[i + 1] * m_shape[i + 1];
                }
            }
        }
    }

    virtual ~TensorDesc() = default;

    inline DTypeName dtype() const { return m_dtype; }

    inline size_t dim(void) const {
        size_t i;
        for (i = 0; i <= TensorMaxDim; ++i)
            if (m_shape[i] == TensorShape0) break;
        return i;
    }

    inline size_t *shape(void) { return m_shape; }
    inline const size_t *shape(void) const { return m_shape; }
    inline size_t *stride(void) { return m_stride; }
    inline const size_t *stride(void) const { return m_stride; }

    inline size_t shape(int i) const { return m_shape[i]; }
    inline size_t &shape(int i) { return m_shape[i]; }
    inline size_t stride(int i) const { return m_stride[i]; }
    inline size_t &stride(int i) { return m_stride[i]; }

    inline size_t numel(void) const {
        size_t n = 1;
        for (int i = 0; i < TensorMaxDim; ++i) {
            if (m_shape[i] == -1) break;
            n *= m_shape[i];
        }
        return n;
    }

    inline bool is_compatible(const TensorDesc &rhs) {
        if (m_dtype != rhs.m_dtype) return false;
        for (int i = 0; i < TensorMaxDim; ++i) {
            if (m_shape[i] != rhs.m_shape[i]) return false;
        }
        return true;
    }

    inline friend std::ostream &operator << (std::ostream &out, const TensorDesc &desc) {
        size_t d = desc.dim();
        out << "TensorDesc(" << "dim=" << d << ", shape=[";
        for (int i = 0; i < d; ++i) out << desc.m_shape[i] << (i == d - 1 ? "" : ", ");
        out << "], stride=[";
        for (int i = 0; i < d; ++i) out << desc.m_stride[i] << (i == d - 1 ? "" : ", ");
        out << "])";
        return out;
    }

private:
    DTypeName m_dtype;
    size_t m_shape[TensorMaxDim + 1];
    size_t m_stride[TensorMaxDim + 1];
};

template <DTypeName DT>
class TensorStorage {
public:
    using cctype = typename DType<DT>::cctype;

    TensorStorage() : m_data_ptr(nullptr) {}
    explicit TensorStorage(cctype *data_ptr, size_t size) : m_data_ptr(data_ptr), m_size(size) {}
    explicit TensorStorage(size_t size) : m_size(size) {
        /* TODO: use aligned allocation. */
        m_data_ptr = new cctype[size];
    }

    /* NB: delete the copy-constructor and move-constructor */
    TensorStorage(const TensorStorage<DT> &) = delete;
    TensorStorage(TensorStorage<DT> &&) = delete;

    virtual ~TensorStorage() {
        if (m_data_ptr != nullptr) {
            delete []m_data_ptr;
            m_data_ptr = nullptr;
        }
    }

    size_t size() const { return m_size; }
    size_t memsize() const { return m_size * sizeof(cctype); }
    const cctype *data_ptr() const { return m_data_ptr; }
    cctype *mutable_data_ptr() { return m_data_ptr; }

    inline friend std::ostream &operator << (std::ostream &out, const TensorStorage<DT> &storage) {
        out << "TensorStorage(dtype=" << DType<DT>::name << ", " << "size=" << storage.m_size << ", data_ptr=" << storage.m_data_ptr  << ")";
        return out;
    }

private:
    cctype *m_data_ptr;
    size_t m_size;
};

template <DTypeName DT>
class TensorImpl;

class Tensor {
public:
    Tensor() : m_desc() {}
    Tensor(const TensorDesc &desc) : m_desc(desc) {}
    virtual ~Tensor() = default;

    inline TensorDesc &desc(void) {
        return m_desc;
    }
    inline const TensorDesc &desc(void) const {
        return m_desc;
    }
    template <DTypeName DT>
    inline TensorImpl<DT> &as(void) {
        return *(dynamic_cast<TensorImpl<DT> *>(this));
    }
    template <DTypeName DT>
    inline const TensorImpl<DT> &as(void) const {
        return *(dynamic_cast<TensorImpl<DT> *>(this));
    }

protected:
    TensorDesc m_desc;
};

typedef std::shared_ptr<Tensor> TensorPtr;

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

    inline const TensorStorage<DT> &storage() const { return *m_storage; }
    inline TensorStorage<DT> &storage() { return *m_storage; }

    inline const cctype *data_ptr() const {
        return m_storage->data_ptr();
    }
    inline cctype *mutable_data_ptr() {
        return m_storage.mutable_data_ptr();
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
TensorPtr empty(DTypeName dtype, const std::initializer_list<size_t> &shape);

} /* !namespace ncg */

#endif /* !TENSOR_H */
