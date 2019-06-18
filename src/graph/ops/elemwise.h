/*
 * elemwise.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#pragma once

#include "core/tensor_impl.h"
#include "core/ops/elemwise.h"
#include "graph/op.h"

namespace ncg {

template <typename OpClass>
class GOpElemwiseBase : public GraphOpWrapper<OpClass> {
public:
    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        NCG_OP_CHECK_NONEMPTY_INPUTS(graph, inputs);
        NCG_OP_CHECK_COMPATIBLE_DTYPE(graph, inputs);
        NCG_OP_CHECK_COMPATIBLE_SHAPE(graph, inputs);
    }
};

class GOpCast : public GOpElemwiseBase<OpCast>, public GraphSingleOutputOp {
public:
    NCG_GOP_DEF_NAME(GOpCast);

    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        GOpElemwiseBase<OpCast>::check_inputs(graph, inputs);
        NCG_OP_CHECK_CTX_CLEAN(graph);
        NCG_OP_CHECK_NR_INPUTS(graph, inputs, 1);
    }

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        auto dtype = this->template desc<OpCastDesc>().dtype;
        TensorDesc desc(dtype, inputs[0]->desc().shape_vec());
        return {this->make_tensor(0, desc)};
    }

    virtual void backward(Graph &graph, GTensorPtr loss);
};

class GOpCond : public GOpElemwiseBase<OpCond>, public GraphSingleOutputOp {
public:
    NCG_GOP_DEF_NAME(GOpCond);

    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        GOpElemwiseBase::check_inputs(graph, inputs);
        NCG_OP_CHECK_CTX_CLEAN(graph);
        NCG_OP_CHECK_NR_INPUTS(graph, inputs, 3);
    }

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        return {make_tensor(0, inputs[2]->desc())};
    }

    virtual void backward(Graph &graph, GTensorPtr loss);
};

template <typename OpClass>
class GOpUnaryElemwiseBase : public GOpElemwiseBase<OpClass>, public GraphSingleOutputOp {
public:
    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        GOpElemwiseBase<OpClass>::check_inputs(graph, inputs);
        NCG_OP_CHECK_CTX_CLEAN(graph);
        NCG_OP_CHECK_NR_INPUTS(graph, inputs, 1);
    }

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        TensorDesc desc(inputs[0]->desc().dtype(), inputs[0]->desc().shape_vec());
        return {this->make_tensor(0, desc)};
    }
};

template <typename OpClass>
class GOpBinaryElemwiseBase : public GOpElemwiseBase<OpClass>, public GraphSingleOutputOp {
public:
    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        GOpElemwiseBase<OpClass>::check_inputs(graph, inputs);
        NCG_OP_CHECK_CTX_CLEAN(graph);
        NCG_OP_CHECK_NR_INPUTS(graph, inputs, 2);
    }

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        TensorDesc desc(inputs[0]->desc().dtype(), inputs[0]->desc().shape_vec());
        return {this->make_tensor(0, desc)};
    }
};

#define DEF_GOP_UNARY(name) \
class GOp##name : public GOpUnaryElemwiseBase<Op##name> { \
public: \
    NCG_GOP_DEF_NAME(GOp##name); \
    virtual void backward(Graph &graph, GTensorPtr loss); \
}

DEF_GOP_UNARY(Neg);
DEF_GOP_UNARY(Sin);
DEF_GOP_UNARY(Cos);
DEF_GOP_UNARY(Tan);
DEF_GOP_UNARY(Log);
DEF_GOP_UNARY(Exp);
DEF_GOP_UNARY(Tanh);
DEF_GOP_UNARY(Sigmoid);
DEF_GOP_UNARY(Reciprocal);

#undef DEF_GOP_UNARY

#define DEF_GOP_BINARY(name) \
class GOp##name : public GOpBinaryElemwiseBase<Op##name> { \
public: \
    NCG_GOP_DEF_NAME(GOp##name); \
    virtual void backward(Graph &graph, GTensorPtr loss); \
}

DEF_GOP_BINARY(Add);
DEF_GOP_BINARY(Sub);
DEF_GOP_BINARY(Mul);
DEF_GOP_BINARY(Div);
DEF_GOP_BINARY(Ge);
DEF_GOP_BINARY(Le);
DEF_GOP_BINARY(Geq);
DEF_GOP_BINARY(Leq);
DEF_GOP_BINARY(Eq);
DEF_GOP_BINARY(Neq);
DEF_GOP_BINARY(Pow);
DEF_GOP_BINARY(Min);
DEF_GOP_BINARY(Max);

#undef DEF_GOP_BINARY

} /* !namespace ncg */

