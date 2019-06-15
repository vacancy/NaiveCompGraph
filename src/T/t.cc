/*
 * t.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#define NCG_T_DEF_EXTERN
#include "T/t.h"
#undef NCG_T_DEF_EXTERN
#include <memory>

namespace ncg {
namespace T {

static ::ncg::DTypeName Int8 = ::ncg::DTypeName::Int8;
static ::ncg::DTypeName Int32 = ::ncg::DTypeName::Int32;
static ::ncg::DTypeName Int64 = ::ncg::DTypeName::Int64;
static ::ncg::DTypeName Float32 = ::ncg::DTypeName::Float32;
static ::ncg::DTypeName Float64 = ::ncg::DTypeName::Float64;

namespace {
static auto default_graph_manager = std::make_unique<DefaultManager<Graph>>(true);
static auto default_session_manager = std::make_unique<DefaultManager<Session>>();
static auto default_session = std::make_unique<Session>(default_graph_manager->get_default());

struct DefaultSessionInitializer {
    DefaultSessionInitializer() {
        default_session_manager->as_default(default_session.get());
    }
};

static DefaultSessionInitializer _default_session_initializer;
}

void as_default_graph(Graph &graph) {
    default_graph_manager->as_default(&graph);
}

Graph &get_default_graph() {
    return default_graph_manager->get_default();
}

void restore_default_graph() {
    default_graph_manager->restore_default();
}

void as_default_session(Session &session) {
    default_session_manager->as_default(&session);
}

Session &get_default_session() {
    return default_session_manager->get_default();
}

void restore_default_session() {
    default_session_manager->restore_default();
}

TTensor::TTensor(const TensorPtr &value) : TTensor(as_tensor(value)) {}

TTensor TTensor::eq(const TTensor &rhs) const {
    return ::ncg::T::eq(*this, rhs);
}

TTensor TTensor::neq(const TTensor &rhs) const {
    return ::ncg::T::neq(*this, rhs);
}

#define NCG_T_DEF_OPERATOR(op_symbol, op_name) TTensor operator op_symbol (const TTensor &a, const TTensor &b) { return op_name(a, b); }

NCG_T_DEF_OPERATOR(+, add);
NCG_T_DEF_OPERATOR(-, sub);
NCG_T_DEF_OPERATOR(*, mul);
NCG_T_DEF_OPERATOR(/, div);
NCG_T_DEF_OPERATOR(>, ge);
NCG_T_DEF_OPERATOR(<, le);
NCG_T_DEF_OPERATOR(>=, geq);
NCG_T_DEF_OPERATOR(<=, leq);

TTensor as_tensor(const TTensor &value) {
    return value;
}

TTensor as_tensor(const GTensorPtr &value) {
    return TTensor(value);
}

TTensor as_tensor(const TensorPtr &value) {
    return constant(value);
}

#define NCG_T_DEF_UNARY(op_name, gop_name) TTensor op_name(TTensor a) { \
    Graph &g = get_default_graph(); \
    return g.op<GOp##gop_name>(nullptr, a); \
}

NCG_T_DEF_UNARY(neg, Neg);
NCG_T_DEF_UNARY(sin, Sin);
NCG_T_DEF_UNARY(cos, Cos);
NCG_T_DEF_UNARY(tan, Tan);
NCG_T_DEF_UNARY(log, Log);
NCG_T_DEF_UNARY(exp, Exp);
NCG_T_DEF_UNARY(tanh, Tanh);
NCG_T_DEF_UNARY(sigmoid, Sigmoid);
NCG_T_DEF_UNARY(reciprocal, Reciprocal);

#define NCG_T_DEF_BINARY(op_name, gop_name) TTensor op_name(TTensor a, TTensor b) { \
    Graph &g = get_default_graph(); \
    return g.op<GOp##gop_name>(nullptr, a, b); \
}

NCG_T_DEF_BINARY(add, Add);
NCG_T_DEF_BINARY(sub, Sub);
NCG_T_DEF_BINARY(mul, Mul);
NCG_T_DEF_BINARY(div, Div);
NCG_T_DEF_BINARY(ge, Ge);
NCG_T_DEF_BINARY(le, Le);
NCG_T_DEF_BINARY(geq, Geq);
NCG_T_DEF_BINARY(leq, Leq);
NCG_T_DEF_BINARY(eq, Eq);
NCG_T_DEF_BINARY(neq, Neq);
NCG_T_DEF_BINARY(pow, Pow);

TTensor placeholder(std::string name, const ShapeVec &shape, DTypeName dtype) {
    Graph &g = get_default_graph();
    return g.op<GOpPlaceholder>(name, OpDescPtr(new ::ncg::GOpPlaceholderDesc(dtype, shape)));
}

TTensor constant(TensorPtr value) {
    Graph &g = get_default_graph();
    return g.op<GOpConstant>(OpDescPtr(new ::ncg::GOpConstantDesc(value)));
}

TTensor variable(std::string name, TensorPtr init_value) {
    Graph &g = get_default_graph();
    return g.op<GOpVariable>(name, OpDescPtr(new ::ncg::GOpVariableDesc(init_value)));
}

TTensor zeros(const ShapeVec &shape, DTypeName dtype) {
    Graph &g = get_default_graph();
    return g.op<GOpZeros>(OpDescPtr(new ::ncg::OpZerosDesc(dtype, shape)));
}

TTensor ones(const ShapeVec &shape, DTypeName dtype) {
    Graph &g = get_default_graph();
    return g.op<GOpOnes>(OpDescPtr(new ::ncg::OpOnesDesc(dtype, shape)));
}

TTensor matmul(TTensor a, TTensor b, bool transpose_a, bool transpose_b) {
    Graph &g = get_default_graph();
    return g.op<GOpMatMul>(OpDescPtr(new ::ncg::OpMatMulDesc(transpose_a, transpose_b)), a, b);
}

TTensor assign(TTensor a, TTensor b) {
    Graph &g = get_default_graph();
    return g.op<GOpAssign>(nullptr, a, b);
}

GraphForwardContext forward_ctx() {
    return GraphForwardContext(get_default_session());
}

} /* !namespace T */
} /* !namespace ncg */

