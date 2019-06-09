/*
 * linalg.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#ifndef LINALG_H
#define LINALG_H

#include "core/op.h"

namespace ncg {

class OpMatmulDesc : public OpDesc {
public:
    OpMatmulDesc() : transpose_a(false), transpose_b(false) {}
    OpMatmulDesc(bool transpose_a, bool transpose_b) : transpose_a(transpose_a), transpose_b(transpose_b) {}
    virtual ~OpMatmulDesc() = default;

    bool transpose_a, transpose_b;
};

class OpMatmul : public Op {
public:
    NCG_DEF_OPNAME(OpMatmul);

    virtual void check_inputs(OpContext &ctx, const TensorVec &inputs) {
        NCG_OP_CHECK_NR_INPUTS(ctx, inputs, 2);
        NCG_OP_CHECK_COMPATIBLE_DTYPE(ctx, inputs);
        NCG_OP_CHECK_INPUT_DIM(ctx, inputs, 0, 2);
        NCG_OP_CHECK_INPUT_DIM(ctx, inputs, 1, 2);

        if (inputs[0]->desc().shape(1) != inputs[1]->desc().shape(0)) {
            ctx.error(this) << "Invalid shape: " << inputs[0]->desc().shape_vec() << " vs. " << inputs[1]->desc().shape_vec() << ".";
        }
    }

    virtual TensorVec compute(OpContext &ctx, const TensorVec &inputs) {
        auto output = zeros(inputs[0]->desc().dtype(), {inputs[0]->desc().shape(0), inputs[1]->desc().shape(1)});
#define MATMUL_DTYPE_CASE(dtype_name) kernel_(ctx, inputs[0]->as<DTypeName::dtype_name>(), inputs[1]->as<DTypeName::dtype_name>(), output->as<DTypeName::dtype_name>());
NCG_SWITCH_DTYPE_ALL(inputs[0]->desc().dtype(), MATMUL_DTYPE_CASE);
#undef MATMUL_DTYPE_CASE

        return {output};
    }

private:
    template<DTypeName DT>
    void kernel_(OpContext &ctx, TensorImpl<DT> *a, TensorImpl<DT> *b, TensorImpl<DT> *c) {
        const auto &desc = this->template desc<OpMatmulDesc>();

        a->make_contiguous();
        b->make_contiguous();
        c->make_contiguous();

        ssize_t N = a->desc().shape(0);
        ssize_t M = b->desc().shape(1);
        ssize_t K = a->desc().shape(1);

        if (!desc.transpose_a && !desc.transpose_b) {
            for (ssize_t k = 0; k < K; ++k) {
                for (ssize_t i = 0; i < N; ++i) {
                    for (ssize_t j = 0; j < M; ++j) {
                        c->mutable_data_ptr()[i * M + j] += a->data_ptr()[i * K + j] * a->data_ptr()[j * M + k];
                    }
                }
            }
        } else {
            ctx.error(this) << "Not implemented transposed MatMul.";
        }
    }
};

} /* !namespace ncg */

#endif /* !LINALG_H */
