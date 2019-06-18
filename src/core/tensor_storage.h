/*
 * tensor_storage.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#pragma once

#include "core/common.h"
#include "core/datatype.h"
#include <limits>

namespace ncg {

const size_t TensorValueMaxPrint = 16;

class TensorStorage {
public:
    TensorStorage(DTypeName dtype);
    virtual ~TensorStorage() = default;

    DTypeName dtype() const;
    virtual size_t size() const = 0;
    virtual size_t memsize() const = 0;

    virtual TensorStorage *clone(ssize_t start = 0, ssize_t length = std::numeric_limits<ssize_t>::max()) const = 0;

    friend std::ostream &operator << (std::ostream &out, const TensorStorage &storage);

protected:
    DTypeName m_dtype;
};

template <DTypeName DT> class TensorStorageImpl;
template <DTypeName DT> std::ostream &operator << (std::ostream &out, const TensorStorageImpl<DT> &storage);

template <DTypeName DT>
class TensorStorageImpl : public TensorStorage {
public:
    using cctype = typename DType<DT>::cctype;

    TensorStorageImpl();
    explicit TensorStorageImpl(cctype *data_ptr, size_t size);
    explicit TensorStorageImpl(size_t size);

    /* NB: delete the copy-constructor and move-constructor */
    TensorStorageImpl(const TensorStorageImpl<DT> &) = delete;
    TensorStorageImpl(TensorStorageImpl<DT> &&) = delete;

    virtual ~TensorStorageImpl();

    virtual size_t size() const;
    virtual size_t memsize() const;

    const cctype *data_ptr() const;
    cctype *mutable_data_ptr();

    virtual TensorStorage *clone(ssize_t start = 0, ssize_t length = std::numeric_limits<ssize_t>::max()) const;

    friend std::ostream &operator << <> (std::ostream &out, const TensorStorageImpl<DT> &storage);

protected:
    cctype *m_data_ptr;
    size_t m_size;
};


} /* !namespace ncg */

