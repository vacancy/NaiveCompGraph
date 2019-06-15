/*
 * linalg.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#ifndef CORE_OPS_LINALG_H
#define CORE_OPS_LINALG_H

#include "core/op.h"

namespace ncg {

class OpMatMulDesc : public OpDesc {
public:
    OpMatMulDesc() : transpose_a(false), transpose_b(false) {}
    OpMatMulDesc(bool transpose_a, bool transpose_b) : transpose_a(transpose_a), transpose_b(transpose_b) {}
    virtual ~OpMatMulDesc() = default;

    bool transpose_a, transpose_b;
};

class OpMatMul : public Op {
public:
    NCG_OP_DEF_NAME(OpMatMul);

    virtual void check_inputs(OpContext &ctx, const TensorVec &inputs) {
        NCG_OP_CHECK_NR_INPUTS(ctx, inputs, 2);
        NCG_OP_CHECK_COMPATIBLE_DTYPE(ctx, inputs);
        NCG_OP_CHECK_INPUT_DIM(ctx, inputs, 0, 2);
        NCG_OP_CHECK_INPUT_DIM(ctx, inputs, 1, 2);

        const auto &desc = this->template desc<OpMatMulDesc>();
        ssize_t k1 = !desc.transpose_a ? inputs[0]->desc().shape(1) : inputs[0]->desc().shape(0);
        ssize_t k2 = !desc.transpose_b ? inputs[1]->desc().shape(0) : inputs[1]->desc().shape(1);
        if (k1 != k2) {
            ctx.error(this) << "Invalid shape: " << inputs[0]->desc().shape_vec() << " vs. " << inputs[1]->desc().shape_vec() << ".";
        }
    }

    virtual TensorVec compute(OpContext &ctx, const TensorVec &inputs) {
        const auto &desc = this->template desc<OpMatMulDesc>();
        const auto &a = inputs[0], &b = inputs[1];
        ssize_t N = !desc.transpose_a ? a->desc().shape(0) : a->desc().shape(1);
        ssize_t M = !desc.transpose_b ? b->desc().shape(1) : b->desc().shape(0);

        auto output = zeros(inputs[0]->desc().dtype(), {N, M});
#define MATMUL_DTYPE_CASE(dtype_name) kernel_(ctx, inputs[0]->template as<DTypeName::dtype_name>(), inputs[1]->template as<DTypeName::dtype_name>(), output->template as<DTypeName::dtype_name>());
NCG_DTYPE_SWITCH_ALL(inputs[0]->desc().dtype(), MATMUL_DTYPE_CASE);
#undef MATMUL_DTYPE_CASE

        return {output};
    }

private:
    template<DTypeName DT>
    void kernel_(OpContext &ctx, TensorImpl<DT> *a, TensorImpl<DT> *b, TensorImpl<DT> *c) {
        const auto &desc = this->template desc<OpMatMulDesc>();

        a->make_contiguous();
        b->make_contiguous();
        c->make_contiguous();

        ssize_t N = !desc.transpose_a ? a->desc().shape(0) : a->desc().shape(1);
        ssize_t M = !desc.transpose_b ? b->desc().shape(1) : b->desc().shape(0);
        ssize_t K = !desc.transpose_a ? a->desc().shape(1) : a->desc().shape(0);

        if (!desc.transpose_a && !desc.transpose_b) {
            for (ssize_t k = 0; k < K; ++k) {
                for (ssize_t i = 0; i < N; ++i) {
                    for (ssize_t j = 0; j < M; ++j) {
                        c->mutable_data_ptr()[i * M + j] += a->data_ptr()[i * K + k] * b->data_ptr()[k * M + j];
                    }
                }
            }
        } else if (!desc.transpose_a && desc.transpose_b) {
            for (ssize_t i = 0; i < N; ++i) {
                for (ssize_t j = 0; j < M; ++j) {
                    for (ssize_t k = 0; k < K; ++k) {
                        c->mutable_data_ptr()[i * M + j] += a->data_ptr()[i * K + k] * b->data_ptr()[j * K + k];
                    }
                }
            }
        } else if (desc.transpose_a && !desc.transpose_b) {
            for (ssize_t k = 0; k < K; ++k) {
                for (ssize_t i = 0; i < N; ++i) {
                    for (ssize_t j = 0; j < M; ++j) {
                        c->mutable_data_ptr()[i * M + j] += a->data_ptr()[k * N + i] * b->data_ptr()[k * M + j];
                    }
                }
            }
        } else if (desc.transpose_a && desc.transpose_b) {
            for (ssize_t k = 0; k < K; ++k) {
                for (ssize_t i = 0; i < N; ++i) {
                    for (ssize_t j = 0; j < M; ++j) {
                        c->mutable_data_ptr()[i * M + j] += a->data_ptr()[k * N + i] * b->data_ptr()[j * K + k];
                    }
                }
            }
        }
    }
};

} /* !namespace ncg */

#endif /* !CORE_OPS_LINALG_H */
