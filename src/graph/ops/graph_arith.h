/*
 * graph_arith.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#ifndef GRAPH_ARITH_H
#define GRAPH_ARITH_H

#include "ops/arith.h"
#include "graph/ops/graph_op_common.h"

namespace ncg {

#define DEF_UNARY_GOP(name) \
class GOp##name : public GraphBinaryElemWiseOp<Op##name> { \
public: \
    NCG_DEF_GOPNAME(GOp##name) \
}

DEF_UNARY_GOP(Neg);
DEF_UNARY_GOP(Sin);
DEF_UNARY_GOP(Cos);
DEF_UNARY_GOP(Tan);
DEF_UNARY_GOP(Log);
DEF_UNARY_GOP(Exp);
DEF_UNARY_GOP(Tanh);
DEF_UNARY_GOP(Sigmoid);

#define DEF_BINARY_GOP(name) \
class GOp##name : public GraphBinaryElemWiseOp<Op##name> { \
public: \
    NCG_DEF_GOPNAME(GOp##name) \
}

DEF_BINARY_GOP(Add);
DEF_BINARY_GOP(Sub);
DEF_BINARY_GOP(Mul);
DEF_BINARY_GOP(Div);

} /* !namespace ncg */

#endif /* !GRAPH_ARITH_H */
