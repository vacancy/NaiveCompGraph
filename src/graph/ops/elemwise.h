/*
 * elemwise.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#pragma once

#include "core/ops/elemwise.h"
#include "graph/op.h"

namespace ncg {

template <typename OpClass>
class GraphElemWiseOp : public GraphOpWrapper<OpClass> {
public:
    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        NCG_OP_CHECK_NONEMPTY_INPUTS(graph, inputs);
        NCG_OP_CHECK_COMPATIBLE_DTYPE(graph, inputs);
        NCG_OP_CHECK_COMPATIBLE_SHAPE(graph, inputs);
    }
};

template <typename OpClass>
class GraphUnaryElemWiseOp : public GraphElemWiseOp<OpClass>, public GraphSingleOutputOp {
public:
    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        GraphElemWiseOp<OpClass>::check_inputs(graph, inputs);
        NCG_OP_CHECK_CTX_CLEAN(graph);
        NCG_OP_CHECK_NR_INPUTS(graph, inputs, 1);
    }

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        TensorDesc desc(inputs[0]->desc().dtype(), inputs[0]->desc().shape_vec());
        return {this->make_tensor(0, desc)};
    }

};

template <typename OpClass>
class GraphBinaryElemWiseOp : public GraphElemWiseOp<OpClass>, public GraphSingleOutputOp {
public:
    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        GraphElemWiseOp<OpClass>::check_inputs(graph, inputs);
        NCG_OP_CHECK_CTX_CLEAN(graph);
        NCG_OP_CHECK_NR_INPUTS(graph, inputs, 2);
    }

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        TensorDesc desc(inputs[0]->desc().dtype(), inputs[0]->desc().shape_vec());
        return {this->make_tensor(0, desc)};
    }
};

#define DEF_UNARY_GOP(name) \
class GOp##name : public GraphUnaryElemWiseOp<Op##name> { \
public: \
    NCG_GOP_DEF_NAME(GOp##name); \
    virtual void backward(Graph &graph, GTensorPtr loss); \
}

DEF_UNARY_GOP(Neg);
DEF_UNARY_GOP(Sin);
DEF_UNARY_GOP(Cos);
DEF_UNARY_GOP(Tan);
DEF_UNARY_GOP(Log);
DEF_UNARY_GOP(Exp);
DEF_UNARY_GOP(Tanh);
DEF_UNARY_GOP(Sigmoid);
DEF_UNARY_GOP(Reciprocal);

#undef DEF_UNARY_GOP

#define DEF_BINARY_GOP(name) \
class GOp##name : public GraphBinaryElemWiseOp<Op##name> { \
public: \
    NCG_GOP_DEF_NAME(GOp##name); \
    virtual void backward(Graph &graph, GTensorPtr loss); \
}

DEF_BINARY_GOP(Add);
DEF_BINARY_GOP(Sub);
DEF_BINARY_GOP(Mul);
DEF_BINARY_GOP(Div);
DEF_BINARY_GOP(Ge);
DEF_BINARY_GOP(Le);
DEF_BINARY_GOP(Geq);
DEF_BINARY_GOP(Leq);
DEF_BINARY_GOP(Eq);
DEF_BINARY_GOP(Neq);
DEF_BINARY_GOP(Pow);

#undef DEF_BINARY_GOP

} /* !namespace ncg */

