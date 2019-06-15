/*
 * tensor.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include "core/tensor.h"
#include "core/tensor_impl.h"
#include "core/op.h"
#include "ops/elemwise.h"
#include "ops/linalg.h"
#include "ops/shape.h"
#include "ops/slice.h"

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

std::shared_ptr<TensorStorage> Tensor::storage() {
    return m_storage;
}

std::shared_ptr<const TensorStorage> Tensor::storage() const {
    return m_storage;
}

const ssize_t Tensor::elindex(ssize_t elindex) const {
    ssize_t ret = 0;
    for (ssize_t i = m_desc.dim() - 1; i >= 0; --i) {
        ret += (elindex % m_desc.shape(i)) * m_desc.stride(i);
        elindex /= m_desc.shape(i);
    }

    return ret;
}

bool Tensor::own_data() const {
    return m_own_data;
}

ssize_t Tensor::data_ptr_offset() const {
    return  m_data_ptr_offset;
}

std::ostream &operator << (std::ostream &out, const Tensor &tensor) {
    out << "Tensor(desc=" << tensor.desc() << ", storage=" << *tensor.storage() << ")";
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

#define EMPTY_DTYPE_CASE(dtype_name) return TensorPtr( \
        static_cast<Tensor *>(new TensorImpl<DTypeName::dtype_name>(\
            desc, new TensorStorageImpl<DTypeName::dtype_name>(desc.numel()) \
        )) \
    )
NCG_DTYPE_SWITCH_ALL(dtype, EMPTY_DTYPE_CASE)
#undef EMPTY_DTYPE_CASE

    return TensorPtr(nullptr);
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

#define NCG_OP_DEF_UNARY_FUNC(func_name, op_name) TensorPtr func_name(TensorPtr a) { \
    auto ctx = OpContext(); \
    auto op = Op##op_name(); \
    auto output_vec = op.execute(ctx, {a}); \
    ncg_assert_msg(!ctx.is_error(), ctx.error_str()); \
    return ctx.ok() ? output_vec[0] : nullptr; \
}

NCG_OP_DEF_UNARY_FUNC(neg, Neg);
NCG_OP_DEF_UNARY_FUNC(sin, Sin);
NCG_OP_DEF_UNARY_FUNC(cos, Cos);
NCG_OP_DEF_UNARY_FUNC(tan, Tan);
NCG_OP_DEF_UNARY_FUNC(log, Log);
NCG_OP_DEF_UNARY_FUNC(exp, Exp);
NCG_OP_DEF_UNARY_FUNC(tanh, Tanh);
NCG_OP_DEF_UNARY_FUNC(sigmoid, Sigmoid);
NCG_OP_DEF_UNARY_FUNC(reciprocal, Reciprocal);

#define NCG_OP_DEF_BINARY_FUNC(func_name, op_name) TensorPtr func_name(TensorPtr a, TensorPtr b) { \
    auto ctx = OpContext(); \
    auto op = Op##op_name(); \
    auto output_vec = op.execute(ctx, {a, b}); \
    ncg_assert_msg(!ctx.is_error(), ctx.error_str()); \
    return ctx.ok() ? output_vec[0] : nullptr; \
}

NCG_OP_DEF_BINARY_FUNC(add, Add);
NCG_OP_DEF_BINARY_FUNC(sub, Sub);
NCG_OP_DEF_BINARY_FUNC(mul, Mul);
NCG_OP_DEF_BINARY_FUNC(div, Div);
NCG_OP_DEF_BINARY_FUNC(ge, Ge);
NCG_OP_DEF_BINARY_FUNC(le, Le);
NCG_OP_DEF_BINARY_FUNC(geq, Geq);
NCG_OP_DEF_BINARY_FUNC(leq, Leq);
NCG_OP_DEF_BINARY_FUNC(eq, Eq);
NCG_OP_DEF_BINARY_FUNC(neq, Neq);
NCG_OP_DEF_BINARY_FUNC(pow, Pow);

TensorPtr matmul(TensorPtr a, TensorPtr b, bool transpose_a, bool transpose_b) {
    auto ctx = OpContext();
    auto op = OpMatMul();
    op.set_desc(OpDescPtr(new OpMatMulDesc(transpose_a, transpose_b)));
    auto output_vec = op.execute(ctx, {a, b});
    ncg_assert_msg(!ctx.is_error(), ctx.error_str());
    return ctx.ok() ? output_vec[0] : nullptr;
}

#define NCG_OP_DEF_SHAPE_FUNC(func_name, op_name) TensorPtr func_name(TensorPtr a, const ShapeVec &b) { \
    auto ctx = OpContext(); \
    auto op = Op##op_name(); \
    op.set_desc(OpDescPtr(new Op##op_name##Desc(b))); \
    auto output_vec = op.execute(ctx, {a}); \
    ncg_assert_msg(!ctx.is_error(), ctx.error_str()); \
    return ctx.ok() ? output_vec[0] : nullptr; \
}

NCG_OP_DEF_SHAPE_FUNC(reshape, Reshape);
NCG_OP_DEF_SHAPE_FUNC(permute, Permute);
NCG_OP_DEF_SHAPE_FUNC(expand, Expand);

TensorPtr concat(const TensorVec &a, ssize_t axis) {
    auto ctx = OpContext();
    auto op = OpConcat();
    op.set_desc(OpDescPtr(new OpConcatDesc(axis)));
    auto output_vec = op.execute(ctx, a);
    ncg_assert_msg(!ctx.is_error(), ctx.error_str());
    return ctx.ok() ? output_vec[0] : nullptr;
}

TensorVec split(TensorPtr a, ssize_t axis, const ShapeVec &splits) {
    auto ctx = OpContext();
    auto op = OpSplit();
    op.set_desc(OpDescPtr(new OpSplitDesc(axis, splits)));
    auto output_vec = op.execute(ctx, {a});
    ncg_assert_msg(!ctx.is_error(), ctx.error_str());
    return output_vec;
}

TensorPtr narrow(TensorPtr a, ssize_t axis, ssize_t start, ssize_t length) {
    auto ctx = OpContext();
    auto op = OpNarrow();
    op.set_desc(OpDescPtr(new OpNarrowDesc(axis, start, length)));
    auto output_vec = op.execute(ctx, {a});
    ncg_assert_msg(!ctx.is_error(), ctx.error_str());
    return ctx.ok() ? output_vec[0] : nullptr;
}

TensorPtr index_select(TensorPtr a, ssize_t axis, TensorPtr b) {
    auto ctx = OpContext();
    auto op = OpIndexSelect();
    op.set_desc(OpDescPtr(new OpIndexSelectDesc(axis)));
    auto output_vec = op.execute(ctx, {a, b});
    ncg_assert_msg(!ctx.is_error(), ctx.error_str());
    return ctx.ok() ? output_vec[0] : nullptr;
}

TensorPtr gather(TensorPtr a, ssize_t axis, TensorPtr b) {
    auto ctx = OpContext();
    auto op = OpGather();
    op.set_desc(OpDescPtr(new OpGatherDesc(axis)));
    auto output_vec = op.execute(ctx, {a, b});
    ncg_assert_msg(!ctx.is_error(), ctx.error_str());
    return ctx.ok() ? output_vec[0] : nullptr;
}

#define NCG_OP_DEF_OPERATOR_FUNC(op_symbol, op_func) TensorPtr operator op_symbol (const TensorPtr &a, const TensorPtr &b) { \
    return op_func(a, b); \
}

NCG_OP_DEF_OPERATOR_FUNC(+, add);
NCG_OP_DEF_OPERATOR_FUNC(-, sub);
NCG_OP_DEF_OPERATOR_FUNC(*, mul);
NCG_OP_DEF_OPERATOR_FUNC(/, div);
NCG_OP_DEF_OPERATOR_FUNC(>, ge);
NCG_OP_DEF_OPERATOR_FUNC(<, le);
NCG_OP_DEF_OPERATOR_FUNC(>=, geq);
NCG_OP_DEF_OPERATOR_FUNC(<=, leq);

TensorPtr TensorPtr::eq(const TensorPtr &rhs) const {
    return ::ncg::eq(*this, rhs);
}

TensorPtr TensorPtr::neq(const TensorPtr &rhs) const {
    return ::ncg::neq(*this, rhs);
}

TensorPtr TensorPtr::reshape(const ShapeVec &shape) const {
    return ::ncg::reshape(*this, shape);
}

TensorPtr TensorPtr::permute(const ShapeVec &axes) const {
    return ::ncg::permute(*this, axes);
}

TensorPtr TensorPtr::expand(const ShapeVec &shape) const {
    return ::ncg::expand(*this, shape);
}

TensorPtr TensorPtr::narrow(ssize_t axis, ssize_t start, ssize_t length) {
    return ::ncg::narrow(*this, axis, start, length);
}

TensorPtr TensorPtr::index_select(ssize_t axis, const TensorPtr &indices) {
    return ::ncg::index_select(*this, axis, indices);
}

TensorPtr TensorPtr::gather(ssize_t axis, const TensorPtr &indices) {
    return ::ncg::gather(*this, axis, indices);
}

std::ostream &operator << (std::ostream &out, const TensorPtr &tensor) {
    out << *tensor;
    return out;
}

} /* !namespace ncg */
