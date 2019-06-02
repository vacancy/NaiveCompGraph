/*
 * graph_arith.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#ifndef GRAPH_OPS_ARITH_H
#define GRAPH_OPS_ARITH_H

#include "core/ops/arith.h"
#include "graph/op.h"
#include "graph/ops/op_common.h"

namespace ncg {

#define DEF_UNARY_GOP(name) \
class GOp##name : public GraphUnaryElemWiseOp<Op##name> { \
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
DEF_BINARY_GOP(Ge);
DEF_BINARY_GOP(Le);
DEF_BINARY_GOP(Geq);
DEF_BINARY_GOP(Leq);
DEF_BINARY_GOP(Eq);
DEF_BINARY_GOP(Neq);

} /* !namespace ncg */

#endif /* !GRAPH_OPS_ARITH_H */
