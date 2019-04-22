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

class GOpAdd : public GraphBinaryElemWiseOp<OpAdd> {
public:
    NCG_DEF_GOPNAME(GOpAdd);
};
class GOpSub : public GraphBinaryElemWiseOp<OpSub> {
public:
    NCG_DEF_GOPNAME(GOpSub);
};
class GOpMul : public GraphBinaryElemWiseOp<OpMul> {
public:
    NCG_DEF_GOPNAME(GOpMul);
};
class GOpDiv : public GraphBinaryElemWiseOp<OpDiv> {
public:
    NCG_DEF_GOPNAME(GOpDiv);
};

} /* !namespace ncg */

#endif /* !GRAPH_ARITH_H */
