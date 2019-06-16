/*
 * tensor.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include "graph/tensor.h"
#include "graph/op.h"
#include "graph/ops/elemwise.h"
#include "graph/ops/grad.h"
#include "graph/ops/linalg.h"
#include "graph/ops/netsrc.h"
#include "graph/ops/shape.h"
#include "graph/ops/update.h"

namespace ncg {

GraphTensor::GraphTensor() : m_owner_op(), m_owner_op_index(0), m_desc() {
    // Pass
}

GraphTensor::GraphTensor(GraphOp *owner_op, ssize_t index, const TensorDesc &desc) :
    m_owner_op(owner_op), m_owner_op_index(index), m_desc(desc) {
    // Pass
}

ssize_t GraphTensor::owner_op_index() const {
    return m_owner_op_index;
}

TensorDesc &GraphTensor::desc() {
    return m_desc;
}

const TensorDesc &GraphTensor::desc() const {
    return m_desc;
}

GTensorPtr GraphTensor::grad(GTensorPtr loss) const {
    auto tensor = loss.get();
    std::uintptr_t tpi = reinterpret_cast<std::uintptr_t>(tensor);

    auto it = m_grads.find(tpi);
    if (it == m_grads.end()) {
        return nullptr;
    }
    return it->second;
}

void GraphTensor::set_grad(Graph &graph, GTensorPtr loss, GTensorPtr grad) {
    auto tensor = loss.get();
    std::uintptr_t tpi = reinterpret_cast<std::uintptr_t>(tensor);

    auto it = m_grads.find(tpi);
    if (it == m_grads.end()) {
        m_grads.emplace(tpi, grad);
    } else {
        if (grad != nullptr) {
            if (it->second == nullptr) {
                m_grads[tpi] = grad;
            } else {
                m_grads[tpi] = graph.op<GOpAdd>(nullptr, it->second, grad);
            }
        }
    }
}

std::ostream & operator << (std::ostream &out, const GraphTensor &tensor) {
    out << "GTensor(op=" << tensor.m_owner_op->name() << ", op_type=" << tensor.m_owner_op->op_name() << ", index=" << tensor.m_owner_op_index << ")";
    return out;
}

GTensorPtr::GTensorPtr(const TensorPtr &value) : GTensorPtr(as_gtensor(value)) {}

GTensorPtr GTensorPtr::eq(const GTensorPtr &rhs) const {
    return G::eq(*this, rhs);
}

GTensorPtr GTensorPtr::neq(const GTensorPtr &rhs) const {
    return G::neq(*this, rhs);
}

#define NCG_GOP_DEF_OPERATOR(op_symbol, op_name) GTensorPtr operator op_symbol (const GTensorPtr &a, const GTensorPtr &b) { return G::op_name(a, b); }

NCG_GOP_DEF_OPERATOR(+, add);
NCG_GOP_DEF_OPERATOR(-, sub);
NCG_GOP_DEF_OPERATOR(*, mul);
NCG_GOP_DEF_OPERATOR(/, div);
NCG_GOP_DEF_OPERATOR(>, ge);
NCG_GOP_DEF_OPERATOR(<, le);
NCG_GOP_DEF_OPERATOR(>=, geq);
NCG_GOP_DEF_OPERATOR(<=, leq);

GTensorPtr as_gtensor(const GTensorPtr &value) {
    return value;
}

GTensorPtr as_gtensor(const TensorPtr &value) {
    return G::constant(value);
}

namespace G {

#define NCG_GOP_DEF_UNRAY_FUNC(op_name, gop_name) GTensorPtr op_name(GTensorPtr a) { \
    Graph &g = get_default_graph(); \
    return g.op<GOp##gop_name>(nullptr, a); \
}

NCG_GOP_DEF_UNRAY_FUNC(neg, Neg);
NCG_GOP_DEF_UNRAY_FUNC(sin, Sin);
NCG_GOP_DEF_UNRAY_FUNC(cos, Cos);
NCG_GOP_DEF_UNRAY_FUNC(tan, Tan);
NCG_GOP_DEF_UNRAY_FUNC(log, Log);
NCG_GOP_DEF_UNRAY_FUNC(exp, Exp);
NCG_GOP_DEF_UNRAY_FUNC(tanh, Tanh);
NCG_GOP_DEF_UNRAY_FUNC(sigmoid, Sigmoid);
NCG_GOP_DEF_UNRAY_FUNC(reciprocal, Reciprocal);

#define NCG_GOP_DEF_BINARY_FUNC(op_name, gop_name) GTensorPtr op_name(GTensorPtr a, GTensorPtr b) { \
    Graph &g = get_default_graph(); \
    return g.op<GOp##gop_name>(nullptr, a, b); \
}

NCG_GOP_DEF_BINARY_FUNC(add, Add);
NCG_GOP_DEF_BINARY_FUNC(sub, Sub);
NCG_GOP_DEF_BINARY_FUNC(mul, Mul);
NCG_GOP_DEF_BINARY_FUNC(div, Div);
NCG_GOP_DEF_BINARY_FUNC(ge, Ge);
NCG_GOP_DEF_BINARY_FUNC(le, Le);
NCG_GOP_DEF_BINARY_FUNC(geq, Geq);
NCG_GOP_DEF_BINARY_FUNC(leq, Leq);
NCG_GOP_DEF_BINARY_FUNC(eq, Eq);
NCG_GOP_DEF_BINARY_FUNC(neq, Neq);
NCG_GOP_DEF_BINARY_FUNC(pow, Pow);

GTensorPtr placeholder(std::string name, const ShapeVec &shape, DTypeName dtype) {
    Graph &g = get_default_graph();
    return g.op<GOpPlaceholder>(name, OpDescPtr(new ::ncg::GOpPlaceholderDesc(dtype, shape)));
}

GTensorPtr constant(TensorPtr value) {
    Graph &g = get_default_graph();
    return g.op<GOpConstant>(OpDescPtr(new ::ncg::GOpConstantDesc(value)));
}

GTensorPtr variable(std::string name, TensorPtr init_value) {
    Graph &g = get_default_graph();
    return g.op<GOpVariable>(name, OpDescPtr(new ::ncg::GOpVariableDesc(init_value)));
}

GTensorPtr zeros(const ShapeVec &shape, DTypeName dtype) {
    Graph &g = get_default_graph();
    return g.op<GOpZeros>(OpDescPtr(new ::ncg::OpZerosDesc(dtype, shape)));
}

GTensorPtr ones(const ShapeVec &shape, DTypeName dtype) {
    Graph &g = get_default_graph();
    return g.op<GOpOnes>(OpDescPtr(new ::ncg::OpOnesDesc(dtype, shape)));
}

GTensorPtr matmul(GTensorPtr a, GTensorPtr b, bool transpose_a, bool transpose_b) {
    Graph &g = get_default_graph();
    return g.op<GOpMatMul>(OpDescPtr(new ::ncg::OpMatMulDesc(transpose_a, transpose_b)), a, b);
}

GTensorPtr assign(GTensorPtr a, GTensorPtr b) {
    Graph &g = get_default_graph();
    return g.op<GOpAssign>(nullptr, a, b);
}

GTensorPtr shapeof(GTensorPtr a) {
    Graph &g = get_default_graph();
    return g.op<GOpShapeOf>(nullptr, a);
}

GTensorPtr shapeof(GTensorPtr a, ssize_t axis) {
    Graph &g = get_default_graph();
    return g.op<GOpShapeOfIndex>(OpDescPtr(new ::ncg::OpShapeOfIndexDesc(axis)), a);
}

GTensorPtr shape_cat(GTensorVec a) {
    Graph &g = get_default_graph();
    return g.op<GOpShapeConcat>(nullptr, a);
}

} /* !namespace graph */

} /* !namespace ncg */
