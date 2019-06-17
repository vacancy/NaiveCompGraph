/*
 * linalg.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#pragma once

#include "core/tensor.h"
#include "core/tensor_impl.h"
#include "core/ops/linalg.h"
#include "graph/op.h"

namespace ncg {

class GOpMatMul : public GraphOpWrapper<OpMatMul>, GraphSingleOutputOp {
public:
    NCG_GOP_DEF_NAME(GOPMatMul);

    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        NCG_OP_CHECK_NR_INPUTS(graph, inputs, 2);
        NCG_OP_CHECK_COMPATIBLE_DTYPE(graph, inputs);
        NCG_OP_CHECK_INPUT_DIM(graph, inputs, 0, 2);
        NCG_OP_CHECK_INPUT_DIM(graph, inputs, 1, 2);

        const auto &desc = this->template desc<OpMatMulDesc>();
        ssize_t k1 = !desc.transpose_a ? inputs[0]->desc().shape(1) : inputs[0]->desc().shape(0);
        ssize_t k2 = !desc.transpose_b ? inputs[1]->desc().shape(0) : inputs[1]->desc().shape(1);
        if (k1 != k2) {
            graph.error(this) << "Invalid shape: " << inputs[0]->desc().shape_vec() << " vs. " << inputs[1]->desc().shape_vec() << ".";
        }
    }

    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        const auto &desc = this->template desc<OpMatMulDesc>();
        const auto &a = inputs[0], &b = inputs[1];
        ssize_t N = !desc.transpose_a ? a->desc().shape(0) : a->desc().shape(1);
        ssize_t M = !desc.transpose_b ? b->desc().shape(1) : b->desc().shape(0);

        return {make_tensor(0, TensorDesc(inputs[0]->desc().dtype(), {N, M}))};
    }

    virtual void backward(Graph &graph, GTensorPtr loss);
};

} /* !namespace ncg */

