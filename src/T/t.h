/*
 * t.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#pragma once

#include "core.h"
#include "graph.h"
#include <string>

namespace ncg {
namespace T {

#ifndef NCG_T_DEF_EXTERN
extern ::ncg::DTypeName Int8;
extern ::ncg::DTypeName Int32;
extern ::ncg::DTypeName Int64;
extern ::ncg::DTypeName Float32;
extern ::ncg::DTypeName Float64;
#endif /* NCG_T_DEF_EXTERN */

using ::ncg::DTypeName;
using ::ncg::ShapeVec;
using ::ncg::TensorDesc;

using ::ncg::Tensor;
using ::ncg::TensorPtr;
using ::ncg::TensorVec;
using ::ncg::Op;
using ::ncg::OpDescPtr;

using ::ncg::GraphTensor;
using ::ncg::GTensorPtr;
using ::ncg::GTensorVec;
using ::ncg::GraphOp;

using ::ncg::Graph;
using ::ncg::Session;
using ::ncg::GraphForwardContext;

void as_default_graph(Graph &);
Graph &get_default_graph();
void restore_default_graph();

void as_default_session(Session &);
Session &get_default_session();
void restore_default_session();

class TTensor : public GTensorPtr {
public:
    TTensor(const TTensor &value) : GTensorPtr(value) {}
    TTensor(const GTensorPtr &value) : GTensorPtr(value) {}
    TTensor(const TensorPtr &value);

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    TTensor(T value);
    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    TTensor(std::vector<T> value);
    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    TTensor(std::vector<std::vector<T>> value);

    TTensor eq(const TTensor &rhs) const;
    TTensor neq(const TTensor &rhs) const;
};

TTensor operator + (const TTensor &a, const TTensor &b);
TTensor operator - (const TTensor &a, const TTensor &b);
TTensor operator * (const TTensor &a, const TTensor &b);
TTensor operator / (const TTensor &a, const TTensor &b);
TTensor operator > (const TTensor &a, const TTensor &b);
TTensor operator < (const TTensor &a, const TTensor &b);
TTensor operator >= (const TTensor &a, const TTensor &b);
TTensor operator <= (const TTensor &a, const TTensor &b);

TTensor as_tensor(const TTensor &value);
TTensor as_tensor(const GTensorPtr &value);
TTensor as_tensor(const TensorPtr &value);

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, TTensor>::type
as_tensor(T value) {
    return TTensor(::ncg::fromcc(::ncg::CCType<T>::identifier, value));
}
template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, TTensor>::type
as_tensor(std::vector<T> value) {
    return TTensor(::ncg::fromcc(::ncg::CCType<T>::identifier, value));
}
template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, TTensor>::type
as_tensor(std::vector<std::vector<T>> value) {
    return TTensor(::ncg::fromcc(::ncg::CCType<T>::identifier, value));
}

template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type*>
TTensor::TTensor(T value) : TTensor(as_tensor(value)) {}
template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type*>
TTensor::TTensor(std::vector<T> value) : TTensor(as_tensor(value)) {}
template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type*>
TTensor::TTensor(std::vector<std::vector<T>> value) : TTensor(as_tensor(value)) {}

TTensor neg(TTensor a);
TTensor sin(TTensor a);
TTensor cos(TTensor a);
TTensor tan(TTensor a);
TTensor log(TTensor a);
TTensor exp(TTensor a);
TTensor tanh(TTensor a);
TTensor sigmoid(TTensor a);
TTensor reciprocal(TTensor a);

TTensor add(TTensor a, TTensor b);
TTensor sub(TTensor a, TTensor b);
TTensor mul(TTensor a, TTensor b);
TTensor div(TTensor a, TTensor b);
TTensor ge(TTensor a, TTensor b);
TTensor le(TTensor a, TTensor b);
TTensor geq(TTensor a, TTensor b);
TTensor leq(TTensor a, TTensor b);
TTensor eq(TTensor a, TTensor b);
TTensor neq(TTensor a, TTensor b);
TTensor pow(TTensor a, TTensor b);

TTensor placeholder(std::string name, const ShapeVec &shape, DTypeName dtype=DTypeName::Float32);
TTensor constant(TensorPtr value);
TTensor variable(std::string name, TensorPtr init_value);
TTensor zeros(const ShapeVec &shape, DTypeName dtype=DTypeName::Float32);
TTensor ones(const ShapeVec &shape, DTypeName dtype=DTypeName::Float32);

TTensor matmul(TTensor a, TTensor b, bool transpose_a=false, bool transpose_b=false);

TTensor assign(TTensor a, TTensor b);

GraphForwardContext forward_ctx();

} /* !namespace T */
} /* !namespace ncg */

