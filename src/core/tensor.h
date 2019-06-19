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
#include "core/pickle.h"

#include <algorithm>
#include <limits>

namespace ncg {

template <DTypeName DT>
class TensorImpl;

class Tensor {
public:
    Tensor();
    Tensor(const TensorDesc &desc, std::shared_ptr<TensorStorage> storage, bool own_data=true, ssize_t data_ptr_offset=0);

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    Tensor(T value);
    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    Tensor(std::vector<T> value);
    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    Tensor(std::vector<std::vector<T>> value);

    virtual ~Tensor() = default;

    void pickle(NCGPickler &pickler) const;

    TensorDesc &desc();
    const TensorDesc &desc() const;

    template <DTypeName DT> TensorImpl<DT> *as();
    template <DTypeName DT> const TensorImpl<DT> *as() const;

    std::shared_ptr<TensorStorage> storage();
    std::shared_ptr<const TensorStorage> storage() const;

    template <typename ...Ints>
    const ssize_t index(Ints... args) const;
    const ssize_t elindex(ssize_t elindex) const;

    bool own_data() const;
    ssize_t data_ptr_offset() const;
    virtual void make_own_data() = 0;
    virtual void make_contiguous() = 0;

    friend std::ostream &operator << (std::ostream &out, const Tensor &tensor);

protected:
    TensorDesc m_desc;
    std::shared_ptr<TensorStorage> m_storage;
    bool m_own_data;
    ssize_t m_data_ptr_offset;
};

class TensorPtr : public std::shared_ptr<Tensor> {
public:
    using std::shared_ptr<Tensor>::shared_ptr;
    using super = std::shared_ptr<Tensor>;

    TensorPtr() : super(nullptr) {}
    TensorPtr(const TensorPtr &other) : super(other) {}

    TensorPtr(const std::shared_ptr<Tensor> &other) : super(other) {}
    TensorPtr(std::shared_ptr<Tensor> &&other) : super(std::forward<std::shared_ptr<Tensor>>(other)) {}

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    TensorPtr(T value);
    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    TensorPtr(std::vector<T> value);
    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    TensorPtr(std::vector<std::vector<T>> value);

    TensorPtr eq(const TensorPtr &rhs) const;
    TensorPtr neq(const TensorPtr &rhs) const;

    TensorPtr cast(DTypeName dtype) const;
    TensorPtr int8() const;
    TensorPtr uint8() const;
    TensorPtr int32() const;
    TensorPtr uint32() const;
    TensorPtr int64() const;
    TensorPtr uint64() const;
    TensorPtr float32() const;
    TensorPtr float64() const;

    std::vector<TensorPtr> min(ssize_t axis, bool keepdims=false) const;
    std::vector<TensorPtr> max(ssize_t axis, bool keepdims=false) const;
    TensorPtr sum(ssize_t axis, bool keepdims=false) const;
    TensorPtr mean(ssize_t axis, bool keepdims=false) const;

    TensorPtr reshape(const ShapeVec &shape) const;
    TensorPtr permute(const ShapeVec &axes) const;
    TensorPtr expand(const ShapeVec &shape) const;
    TensorPtr squeeze(ssize_t axis) const;
    TensorPtr unsqueeze(ssize_t axis) const;

    TensorPtr narrow(ssize_t axis, ssize_t start, ssize_t length) const;
    TensorPtr index_select(ssize_t axis, const TensorPtr &indices) const;
    TensorPtr gather(ssize_t axis, const TensorPtr &indices) const;

    friend std::ostream &operator << (std::ostream &out, const TensorPtr &tensor);
};

typedef std::vector<TensorPtr> TensorVec;

template <DTypeName DT>
class TensorImpl : public Tensor {
public:
    using cctype = typename DType<DT>::cctype;

    TensorImpl();
    explicit TensorImpl(const TensorDesc &desc, std::shared_ptr<TensorStorage> storage, bool own_data=true, ssize_t data_ptr_offset=0);
    explicit TensorImpl(const TensorDesc &desc, TensorStorage *storage, bool own_data=true, ssize_t data_ptr_offset=0);
    explicit TensorImpl(const TensorDesc &desc, cctype *data_ptr, bool own_data=true, ssize_t data_ptr_offset=0);
    virtual ~TensorImpl() = default;

    virtual void make_own_data();
    virtual void make_contiguous();

    template <typename ...Ints>
    const cctype &at(Ints... args) const;
    template <typename ...Ints>
    cctype &mutable_at(Ints... args);
    const cctype &elat(ssize_t i) const;
    cctype &mutable_elat(ssize_t i);
    const cctype *data_ptr() const;
    cctype *mutable_data_ptr();

protected:
    const TensorStorageImpl<DT> *storage_impl_() const;
    TensorStorageImpl<DT> *storage_impl_();
};

TensorPtr tensor(NCGUnpickler &unpickler);
TensorPtr tensor(const TensorDesc &desc, std::shared_ptr<TensorStorage> storage, bool own_data=true, ssize_t data_ptr_offset=0);
TensorPtr empty(DTypeName dtype, const ShapeVec &shape);

template <typename ValueT = double>
typename std::enable_if<std::is_arithmetic<ValueT>::value, TensorPtr>::type
fill(DTypeName dtype, const ShapeVec &shape, ValueT value);

TensorPtr zeros(DTypeName dtype, const ShapeVec &shape);
TensorPtr ones(DTypeName dtype, const ShapeVec &shape);

template <typename ValueT = double>
typename std::enable_if<std::is_arithmetic<ValueT>::value, TensorPtr>::type
scalar(DTypeName dtype, ValueT value);

template <typename ValueT = double>
typename std::enable_if<std::is_arithmetic<ValueT>::value, TensorPtr>::type
fromcc(DTypeName dtype, ValueT value);

template <typename ValueT = double>
typename std::enable_if<std::is_arithmetic<ValueT>::value, TensorPtr>::type
fromcc(DTypeName dtype, std::vector<ValueT> values);

template <typename ValueT = double>
typename std::enable_if<std::is_arithmetic<ValueT>::value, TensorPtr>::type
fromcc(DTypeName dtype, std::vector<std::vector<ValueT>> values);

template <typename ValueT=double>
ValueT tocc_scalar(TensorPtr tensor);

template <typename ValueT=double>
std::vector<ValueT> tocc_vector(TensorPtr tensor);

TensorPtr arange(DTypeName dtype, int64_t begin, int64_t end = std::numeric_limits<int64_t>::min(), int64_t step=1);

// elemwise::misc
TensorPtr cast(TensorPtr a, DTypeName dtype);
TensorPtr cond(TensorPtr a, TensorPtr b, TensorPtr c);

// elemwise::unary
TensorPtr neg(TensorPtr a);
TensorPtr sin(TensorPtr a);
TensorPtr cos(TensorPtr a);
TensorPtr tan(TensorPtr a);
TensorPtr log(TensorPtr a);
TensorPtr exp(TensorPtr a);
TensorPtr tanh(TensorPtr a);
TensorPtr sigmoid(TensorPtr a);
TensorPtr reciprocal(TensorPtr a);

// elemwise::binary
TensorPtr add(TensorPtr a, TensorPtr b);
TensorPtr sub(TensorPtr a, TensorPtr b);
TensorPtr mul(TensorPtr a, TensorPtr b);
TensorPtr div(TensorPtr a, TensorPtr b);
TensorPtr ge(TensorPtr a, TensorPtr b);
TensorPtr le(TensorPtr a, TensorPtr b);
TensorPtr geq(TensorPtr a, TensorPtr b);
TensorPtr leq(TensorPtr a, TensorPtr b);
TensorPtr eq(TensorPtr a, TensorPtr b);
TensorPtr neq(TensorPtr a, TensorPtr b);
TensorPtr pow(TensorPtr a, TensorPtr b);
TensorPtr min(TensorPtr a, TensorPtr b);
TensorPtr max(TensorPtr a, TensorPtr b);

// linalg
TensorPtr matmul(TensorPtr a, TensorPtr b, bool transpose_a=false, bool transpose_b=false);

// reduce
TensorVec reduce_min(TensorPtr a, ssize_t axis, bool keepdims=false);
TensorVec reduce_max(TensorPtr a, ssize_t axis, bool keepdims=false);
TensorPtr reduce_sum(TensorPtr a, ssize_t axis, bool keepdims=false);
TensorPtr reduce_mean(TensorPtr a, ssize_t axis, bool keepdims=false);

// shape
TensorPtr reshape(TensorPtr a, const ShapeVec &shape);
TensorPtr permute(TensorPtr a, const ShapeVec &axes);
TensorPtr expand(TensorPtr a, const ShapeVec &shape);
TensorPtr squeeze(TensorPtr a, ssize_t axis);
TensorPtr unsqueeze(TensorPtr a, ssize_t axis);

// slice
TensorPtr concat(const TensorVec &a, ssize_t axis);
TensorVec split(TensorPtr a, ssize_t axis, const ShapeVec &splits);
TensorPtr narrow(TensorPtr a, ssize_t axis, ssize_t start, ssize_t length);
TensorPtr index_select(TensorPtr a, ssize_t axis, TensorPtr b);
TensorPtr gather(TensorPtr a, ssize_t axis, TensorPtr b);

TensorPtr narrow_backward(TensorPtr a, ssize_t axis, ssize_t start, ssize_t input_size);
TensorPtr index_select_backward(TensorPtr a, ssize_t axis, TensorPtr b, ssize_t input_size);
TensorPtr gather_backward(TensorPtr a, ssize_t axis, TensorPtr b, ssize_t input_size);

TensorPtr operator + (const TensorPtr &a, const TensorPtr &b);
TensorPtr operator - (const TensorPtr &a, const TensorPtr &b);
TensorPtr operator * (const TensorPtr &a, const TensorPtr &b);
TensorPtr operator / (const TensorPtr &a, const TensorPtr &b);
TensorPtr operator > (const TensorPtr &a, const TensorPtr &b);
TensorPtr operator < (const TensorPtr &a, const TensorPtr &b);
TensorPtr operator >= (const TensorPtr &a, const TensorPtr &b);
TensorPtr operator <= (const TensorPtr &a, const TensorPtr &b);

TensorPtr as_tensor(const TensorPtr &value);
template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, TensorPtr>::type
as_tensor(T value);
template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, TensorPtr>::type
as_tensor(std::vector<T> value);
template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, TensorPtr>::type
as_tensor(std::vector<std::vector<T>> value);

} /* !namespace ncg */

