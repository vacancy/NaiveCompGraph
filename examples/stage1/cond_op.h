/*
 * cond_op.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#ifndef COND_OP_H
#define COND_OP_H

#include "ops/op_common.h"
#include "graph/ops/graph_op_common.h"

namespace ncg {

class OpCond : public ElemWiseOp {
    NCG_DEF_OPNAME(OpCond);

    virtual void check_inputs(OpContext &ctx, const TensorVec &inputs) {
        ElemWiseOp::check_inputs(ctx, inputs);
        if (ctx.is_error()) return;
        if (inputs.size() != 3) {
            ctx.error(this) << "Cond op takes three inputs";
        }
    }

    virtual TensorVec compute(OpContext &ctx, const TensorVec &inputs) {
        TensorPtr output = empty(inputs[2]->desc().dtype(), inputs[2]->desc().shape_vec());

#define COND_COMPUTE_TYPE(dtype) compute_inner_<DTypeName::dtype>(ctx, inputs, output)
NCG_SWITCH_DTYPE_ALL(inputs[0]->desc().dtype(), COND_COMPUTE_TYPE);
#undef COND_COMPUTE_TYPE

        return {output};
    }

private:
    template <DTypeName DT>
    void compute_inner_(OpContext &ctx, const TensorVec &inputs, TensorPtr &output) {
        size_t n = inputs[0]->desc().numel();
        auto a = inputs[0]->as<DT>();
        auto b = inputs[1]->as<DT>();
        auto c = inputs[2]->as<DT>();
        auto d = output->as<DT>();
        for (ssize_t i = 0; i < n; ++i) {
            d->elat(i) = a->elat(i) > 0 ? b->elat(i) : c->elat(i);
        }
    }
};

class GOpCond : public GraphElemWiseOp<OpCond>, public GraphSingleOutputOp {
public:
    NCG_DEF_GOPNAME(GOpCond);

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        return {make_tensor(0, inputs[2]->desc())};
    }
};

} /* !namespace ncg */

#endif /* !COND_OP_H */
