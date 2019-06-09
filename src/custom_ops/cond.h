/*
 * cond.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#ifndef CUSTOM_OPS_COND_H
#define CUSTOM_OPS_COND_H

#include "ncg.h"

namespace ncg {

class OpCond : public ElemWiseOp {
public:
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
            d->mutable_elat(i) = a->elat(i) > 0 ? b->elat(i) : c->mutable_elat(i);
        }
    }
};

class GOpCond : public GraphElemWiseOp<OpCond>, public GraphSingleOutputOp {
public:
    NCG_DEF_GOPNAME(GOpCond);

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        return {make_tensor(0, inputs[2]->desc())};
    }

    virtual void backward(Graph &graph, GTensorPtr loss) {
        auto output_grad = m_outputs[0]->grad(loss);
        auto zero_grad = graph.op<GOpZeros>(OpDescPtr(new GOpZerosDesc(output_grad->desc().dtype(), output_grad->desc().shape_vec())));

        m_inputs[0]->set_grad(graph, loss, nullptr);
        m_inputs[1]->set_grad(graph, loss, graph.op<GOpCond>(nullptr, m_inputs[0], output_grad, zero_grad));
        m_inputs[2]->set_grad(graph, loss, graph.op<GOpCond>(nullptr, m_inputs[0], zero_grad, output_grad));
    }
};

} /* !namespace ncg */

#endif /* !CUSTOM_OPS_COND_H */
