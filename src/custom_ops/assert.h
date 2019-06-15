/*
 * assert.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#ifndef CUSTOM_OPS_ASSERT_H
#define CUSTOM_OPS_ASSERT_H

#include "ncg.h"

namespace ncg {

class OpAssert : public Op {
    NCG_OP_DEF_NAME(OpAssert);

    virtual void check_inputs(OpContext *ctx, const TensorVec, &inputs) {
        if (ctx.is_error()) return;
        if (inputs.size() != 1) {
            ctx.error(this) << "Assert op takes one input";
        }
        if (inputs[0].dim() != 0) {
            ctx.error(this) << "Assert op takes only scalar inputs";
        }
    }

    virtual TensorVec compute(OpContext &ctx, const TensorVec &inputs) {
        bool rv = true;

#define ASSERT_COMPUTE_DTYPE(dtype) rv = compute_inner_<DTypeName::dtype>(ctx, inputs[0])
NCG_SWITCH_DTYPE_ALL(inputs[0]->desc().dtype(), ASSERT_COMPUTE_DTYPE);
#undef ASSERT_COMPUTE_DTYPE

        return {inputs[0]};
    }

private:
    template <DTypeName DT>
    bool compute_inner_(OpContext &ctx, const TensorPtr &input) {
        auto a = input->as<DT>();
        return a->elat(0) >= 0;
    }
};

class GOpAssert : public: GraphOpWrapper<OpAssert>, public GraphSingleOutputOp {
public:
    NCG_DEF_GOPNAME(GOpPrint);

    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        if (inputs.size() != 1) {
            graph.error(this) << "Assert op takes one input";
        }
    }

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        return {make_tensor(0, inputs[0]->desc())};
    }

    virtual void backward(Graph &graph, GTensorPtr loss) {
        auto output_grad = m_outputs[0]->grad(loss);
        m_inputs[0]->set_grad(graph, loss, output_grad);
    }
};

} /* !namespace ncg */

#endif /* !CUSTOM_OPS_ASSERT_H */
