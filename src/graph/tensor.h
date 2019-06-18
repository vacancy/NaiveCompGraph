/*
 * tensor.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#pragma once

#include "core/tensor.h"

#include <cstdint>
#include <unordered_map>

namespace ncg {

// From graph/op.h
class GraphOp;

// From graph/graph.h
class Graph;

// Forward define the GTensor type.
class GraphTensor;

class GTensorPtr : public std::shared_ptr<GraphTensor> {
public:
    using std::shared_ptr<GraphTensor>::shared_ptr;
    using super = std::shared_ptr<GraphTensor>;

    GTensorPtr() : super(nullptr) {}
    GTensorPtr(const GTensorPtr &other) : super(other) {}
    GTensorPtr(const std::shared_ptr<GraphTensor> &other) : super(other) {}
    GTensorPtr(std::shared_ptr<GraphTensor> &&other) : super(std::forward<super>(other)) {}

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    GTensorPtr(T value);
    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    GTensorPtr(std::vector<T> value);
    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    GTensorPtr(std::vector<std::vector<T>> value);
    GTensorPtr(const TensorPtr &value);

    GTensorPtr eq(const GTensorPtr &rhs) const;
    GTensorPtr neq(const GTensorPtr &rhs) const;

    GTensorPtr cast(DTypeName dtype) const;
    GTensorPtr int8() const;
    GTensorPtr uint8() const;
    GTensorPtr int32() const;
    GTensorPtr uint32() const;
    GTensorPtr int64() const;
    GTensorPtr uint64() const;
    GTensorPtr float32() const;
    GTensorPtr float64() const;

    std::vector<GTensorPtr> min(ssize_t axis, bool keepdims=false) const;
    std::vector<GTensorPtr> max(ssize_t axis, bool keepdims=false) const;
    GTensorPtr sum(ssize_t axis, bool keepdims=false) const;
    GTensorPtr mean(ssize_t axis, bool keepdims=false) const;

    GTensorPtr reshape(const ShapeVec &shape) const;
    GTensorPtr permute(const ShapeVec &axes) const;
    GTensorPtr expand(const ShapeVec &shape) const;
    GTensorPtr squeeze(ssize_t axis) const;
    GTensorPtr unsqueeze(ssize_t axis) const;

    GTensorPtr narrow(ssize_t axis, ssize_t start, ssize_t length) const;
    GTensorPtr index_select(ssize_t axis, const GTensorPtr &indices) const;
    GTensorPtr gather(ssize_t axis, const GTensorPtr &indices) const;

    GTensorPtr shape() const;
    GTensorPtr shape(ssize_t axis) const;

    friend std::ostream &operator << (std::ostream &out, const GTensorPtr &tensor);
};

GTensorPtr operator - (const GTensorPtr &a);
GTensorPtr operator + (const GTensorPtr &a, const GTensorPtr &b);
GTensorPtr operator - (const GTensorPtr &a, const GTensorPtr &b);
GTensorPtr operator * (const GTensorPtr &a, const GTensorPtr &b);
GTensorPtr operator / (const GTensorPtr &a, const GTensorPtr &b);
GTensorPtr operator > (const GTensorPtr &a, const GTensorPtr &b);
GTensorPtr operator < (const GTensorPtr &a, const GTensorPtr &b);
GTensorPtr operator >= (const GTensorPtr &a, const GTensorPtr &b);
GTensorPtr operator <= (const GTensorPtr &a, const GTensorPtr &b);

GTensorPtr as_gtensor(const GTensorPtr &value);
GTensorPtr as_gtensor(const TensorPtr &value);

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, GTensorPtr>::type
as_gtensor(T value) {
    return as_gtensor(fromcc(CCType<T>::identifier, value));
}
template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, GTensorPtr>::type
as_gtensor(std::vector<T> value) {
    return as_gtensor(fromcc(CCType<T>::identifier, value));
}
template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, GTensorPtr>::type
as_gtensor(std::vector<std::vector<T>> value) {
    return as_gtensor(fromcc(CCType<T>::identifier, value));
}

GTensorPtr as_gtensor(Graph &graph, const TensorPtr &value);

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, GTensorPtr>::type
as_gtensor(Graph &graph, T value) {
    return as_gtensor(graph, fromcc(CCType<T>::identifier, value));
}
template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, GTensorPtr>::type
as_gtensor(Graph &graph, std::vector<T> value) {
    return as_gtensor(graph, fromcc(CCType<T>::identifier, value));
}
template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, GTensorPtr>::type
as_gtensor(Graph &graph, std::vector<std::vector<T>> value) {
    return as_gtensor(graph, fromcc(CCType<T>::identifier, value));
}

template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type*>
GTensorPtr::GTensorPtr(T value) : GTensorPtr(as_gtensor(value)) {}
template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type*>
GTensorPtr::GTensorPtr(std::vector<T> value) : GTensorPtr(as_gtensor(value)) {}
template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type*>
GTensorPtr::GTensorPtr(std::vector<std::vector<T>> value) : GTensorPtr(as_gtensor(value)) {}

typedef std::vector<GTensorPtr> GTensorVec;

class GraphTensor {
public:
    GraphTensor();
    GraphTensor(GraphOp *owner_op, ssize_t index, const TensorDesc &desc);
    virtual ~GraphTensor() = default;

    template <typename OpType=GraphOp>
    OpType *owner_op() {
        return dynamic_cast<OpType *>(m_owner_op);
    }
    template <typename OpType=GraphOp>
    const OpType *owner_op() const {
        return dynamic_cast<OpType *>(m_owner_op);
    }

    ssize_t owner_op_index() const;
    TensorDesc &desc();
    const TensorDesc &desc() const;

    GTensorPtr grad(GTensorPtr loss) const;
    void set_grad(Graph &graph, GTensorPtr loss, GTensorPtr grad);

    friend std::ostream & operator << (std::ostream &, const GraphTensor &);

protected:
    GraphOp *m_owner_op;
    ssize_t m_owner_op_index;
    TensorDesc m_desc;

    std::unordered_map<std::uintptr_t, GTensorPtr> m_grads;
};

namespace G {

GTensorVec auto_broadcast(Graph &graph, const GTensorVec &a);
GTensorVec auto_broadcast(const GTensorVec &a);

// elemwise::misc
GTensorPtr cast(TensorPtr a, DTypeName dtype);
GTensorPtr cond(TensorPtr a, TensorPtr b, TensorPtr c);

// elemwise::unary
GTensorPtr neg(GTensorPtr a);
GTensorPtr sin(GTensorPtr a);
GTensorPtr cos(GTensorPtr a);
GTensorPtr tan(GTensorPtr a);
GTensorPtr log(GTensorPtr a);
GTensorPtr exp(GTensorPtr a);
GTensorPtr tanh(GTensorPtr a);
GTensorPtr sigmoid(GTensorPtr a);
GTensorPtr reciprocal(GTensorPtr a);

GTensorPtr add(GTensorPtr a, GTensorPtr b);
GTensorPtr sub(GTensorPtr a, GTensorPtr b);
GTensorPtr mul(GTensorPtr a, GTensorPtr b);
GTensorPtr div(GTensorPtr a, GTensorPtr b);
GTensorPtr ge(GTensorPtr a, GTensorPtr b);
GTensorPtr le(GTensorPtr a, GTensorPtr b);
GTensorPtr geq(GTensorPtr a, GTensorPtr b);
GTensorPtr leq(GTensorPtr a, GTensorPtr b);
GTensorPtr eq(GTensorPtr a, GTensorPtr b);
GTensorPtr neq(GTensorPtr a, GTensorPtr b);
GTensorPtr pow(GTensorPtr a, GTensorPtr b);
GTensorPtr min(GTensorPtr a, GTensorPtr b);
GTensorPtr max(GTensorPtr a, GTensorPtr b);

// netsrc
GTensorPtr placeholder(std::string name, const ShapeVec &shape, DTypeName dtype=DTypeName::Float32);
GTensorPtr constant(TensorPtr value);
GTensorPtr variable(std::string name, TensorPtr init_value);
GTensorPtr zeros(const ShapeVec &shape, DTypeName dtype=DTypeName::Float32);
GTensorPtr ones(const ShapeVec &shape, DTypeName dtype=DTypeName::Float32);

// linalg
GTensorPtr matmul(GTensorPtr a, GTensorPtr b, bool transpose_a=false, bool transpose_b=false);

// update
GTensorPtr assign(GTensorPtr a, GTensorPtr b);

// reduce
GTensorVec reduce_min(GTensorPtr a, ssize_t axis, bool keepdims=false);
GTensorVec reduce_max(GTensorPtr a, ssize_t axis, bool keepdims=false);
GTensorPtr reduce_sum(GTensorPtr a, ssize_t axis, bool keepdims=false);
GTensorPtr reduce_mean(GTensorPtr a, ssize_t axis, bool keepdims=false);

// shape
GTensorPtr reshape(GTensorPtr a, const ShapeVec &shape);
GTensorPtr permute(GTensorPtr a, const ShapeVec &axes);
GTensorPtr expand(GTensorPtr a, const ShapeVec &shape);
GTensorPtr squeeze(GTensorPtr a, ssize_t axis);
GTensorPtr unsqueeze(GTensorPtr a, ssize_t axis);

// shape
GTensorPtr shape_of(GTensorPtr a);
GTensorPtr shape_of(GTensorPtr a, ssize_t axis);
GTensorPtr shape_cat(const GTensorVec &a);

// slice
GTensorPtr concat(const GTensorVec &a, ssize_t axis);
GTensorVec split(GTensorPtr a, ssize_t axis, const ShapeVec &splits);
GTensorPtr narrow(GTensorPtr a, ssize_t axis, ssize_t start, ssize_t length);
GTensorPtr index_select(GTensorPtr a, ssize_t axis, GTensorPtr b);
GTensorPtr gather(GTensorPtr a, ssize_t axis, GTensorPtr b);

} /* !namespace graph */

} /* !namespace ncg */

