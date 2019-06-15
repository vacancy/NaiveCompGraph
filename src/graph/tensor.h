/*
 * tensor.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#pragma once

#include "core/tensor.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>
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
};

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
    return GTensorPtr(fromcc(CCType<T>::identifier, value));
}
template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, GTensorPtr>::type
as_gtensor(std::vector<T> value) {
    return GTensorPtr(fromcc(CCType<T>::identifier, value));
}
template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, GTensorPtr>::type
as_gtensor(std::vector<std::vector<T>> value) {
    return GTensorPtr(fromcc(CCType<T>::identifier, value));
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

GTensorPtr placeholder(std::string name, const ShapeVec &shape, DTypeName dtype=DTypeName::Float32);
GTensorPtr constant(TensorPtr value);
GTensorPtr variable(std::string name, TensorPtr init_value);
GTensorPtr zeros(const ShapeVec &shape, DTypeName dtype=DTypeName::Float32);
GTensorPtr ones(const ShapeVec &shape, DTypeName dtype=DTypeName::Float32);

GTensorPtr matmul(GTensorPtr a, GTensorPtr b, bool transpose_a=false, bool transpose_b=false);

GTensorPtr assign(GTensorPtr a, GTensorPtr b);

} /* !namespace graph */

} /* !namespace ncg */

