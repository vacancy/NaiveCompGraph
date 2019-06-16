/*
 * shape.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#pragma once

#include "core/op.h"

#include <algorithm>
#include <memory>
#include <vector>

namespace ncg {

class OpReshapeDesc : public OpDesc {
public:
    OpReshapeDesc() : shape() {}
    OpReshapeDesc(const ShapeVec &shape) : shape(shape) {}
    virtual ~OpReshapeDesc() = default;

    ShapeVec shape;
};

class OpReshape : public Op {
public:
    NCG_OP_DEF_NAME(OpReshape);

    virtual void check_inputs(OpContext &ctx, const TensorVec &inputs) {
        NCG_OP_CHECK_NR_INPUTS(ctx, inputs, 1);

#define RESHAPE_INVALID_SHAPE do {\
    ctx.error(this) << "Invalid target shape " << shape << "; input tensor shape is " << inputs[0]->desc().shape_vec(); \
    return; \
} while (0)

        auto &shape = this->template desc<OpReshapeDesc>().shape;

        int nr_negative = 0;
        int nr_total = 1;
        for (int i = 0; i < shape.size(); ++i) {
            if (shape[i] == -1) {
                nr_negative += 1;
            } else if (shape[i] == NewAxis) {
                // pass
            } else if (shape[i] > 0) {
                nr_total *= shape[i];
            } else {
                RESHAPE_INVALID_SHAPE;
            }
        }

        if (nr_negative == 1) {
            if (inputs[0]->desc().numel() % nr_total != 0) {
                RESHAPE_INVALID_SHAPE;
            }
        } else if (nr_negative == 0) {
            if (inputs[0]->desc().numel() != nr_total) {
                RESHAPE_INVALID_SHAPE;
            }
        } else {
            RESHAPE_INVALID_SHAPE;
        }

#undef RESHAPE_INVALID_SHAPE
    }

    virtual TensorVec compute(OpContext &ctx, const TensorVec &inputs) {
        const auto input = inputs[0];
        auto shape = this->template desc<OpReshapeDesc>().shape;

        int negative_idx = -1;
        int nr_total = 1;
        for (ssize_t i = 0; i < shape.size(); ++i) {
            if (shape[i] == -1) {
                negative_idx = 0;
            } else if (shape[i] == NewAxis) {
                shape[i] = 1;
            } else if (shape[i] > 0) {
                nr_total *= shape[i];
            }
        }

        if (negative_idx != -1) {
            shape[negative_idx] = input->desc().numel() / nr_total;
        }

        inputs[0]->make_contiguous();
        TensorDesc desc(input->desc().dtype(), shape);
        TensorPtr output = tensor(desc, input->storage(), false);

        return {output};
    }
};

class OpPermuteDesc : public OpDesc {
public:
    OpPermuteDesc() : axes() {}
    OpPermuteDesc(const ShapeVec &axes) : axes(axes) {}
    virtual ~OpPermuteDesc() = default;

    ShapeVec axes;
};

class OpPermute: public Op {
public:
    NCG_OP_DEF_NAME(OpPermute);

    virtual void check_inputs(OpContext &ctx, const TensorVec &inputs) {
        NCG_OP_CHECK_NR_INPUTS(ctx, inputs, 1);

        auto &axes = this->template desc<OpPermuteDesc>().axes;
        std::vector<bool> used(axes.size());
        for (ssize_t i = 0; i < axes.size(); ++i) {
            int j = axes[i];

            if (j >= 0 && j < axes.size() && !used[j]) {
                used[j] = true;
            } else {
                ctx.error(this) << "Invalid permute axes " << axes << ".";
                return;
            }
        }
    }

    virtual TensorVec compute(OpContext &ctx, const TensorVec &inputs) {
        const auto input = inputs[0];
        auto output = tensor(input->desc(), input->storage(), false);
        auto &axes = this->template desc<OpPermuteDesc>().axes;

        auto stride = input->desc().stride();
        for (ssize_t i = 0; i < input->desc().dim(); ++i) {
            output->desc().stride(i) = input->desc().stride(axes[i]);
            output->desc().shape(i) = input->desc().shape(axes[i]);
        }

        return {output};
    }
};

class OpExpandDesc : public OpDesc {
public:
    OpExpandDesc() : shape() {};
    OpExpandDesc(const ShapeVec &shape) : shape(shape) {}
    virtual ~OpExpandDesc() = default;

    ShapeVec shape;
};

class OpExpand : public Op {
public:
    NCG_OP_DEF_NAME(OpExpand);

    virtual void check_inputs(OpContext &ctx, const TensorVec &inputs) {
        NCG_OP_CHECK_NR_INPUTS(ctx, inputs, 1);

        const auto input = inputs[0];
        auto &shape = this->template desc<OpExpandDesc>().shape;

        if (shape.size() != input->desc().dim()) {
            ctx.error(this) << "Expand op cannot change the dimension.";
            return;
        }
        for (ssize_t i = 0; i < shape.size(); ++i) {
            if (shape[i] != input->desc().shape(i) && input->desc().shape(i) != 1) {
                ctx.error(this) << "Invalid target shape " << shape << "; input tensor shape is " << input->desc().shape_vec() << ".";
                return;
            }
        }
    }

    virtual TensorVec compute(OpContext &ctx, const TensorVec &inputs) {
        const auto input = inputs[0];
        auto &shape = this->template desc<OpExpandDesc>().shape;
        auto output = tensor(input->desc(), input->storage(), false);

        for (ssize_t i = 0; i < input->desc().dim(); ++i) {
            if (shape[i] != input->desc().shape(i)) {
                output->desc().shape(i) = shape[i];
                output->desc().stride(i) = 0;
            }
        }

        return {output};
    }
};

class OpSqueezeDesc : public OpDesc {
public:
    OpSqueezeDesc() : axis(0) {};
    OpSqueezeDesc(ssize_t axis) : axis(axis) {}
    virtual ~OpSqueezeDesc() = default;

    ssize_t axis;
};

class OpSqueeze : public Op {
public:
    NCG_OP_DEF_NAME(OpSqueeze);

    virtual void check_inputs(OpContext &ctx, const TensorVec &inputs) {
        NCG_OP_CHECK_NR_INPUTS(ctx, inputs, 1);

        const auto input = inputs[0];
        auto axis = this->template desc<OpSqueezeDesc>().axis;
        if (axis < 0) axis += input->desc().dim();

        NCG_OP_CHECK_INPUT_DIM_GEQ(ctx, inputs, 0, axis);
        if (input->desc().shape(axis) != 1) {
            ctx.error(this) << "Only size 1 dimensions can be squeezed.";
        }
    }

    virtual TensorVec compute(OpContext &ctx, const TensorVec &inputs) {
        const auto input = inputs[0];
        auto axis = this->template desc<OpSqueezeDesc>().axis;
        if (axis < 0) axis += input->desc().dim();
        auto output = tensor(input->desc(), input->storage(), false);

        auto shape_vec = input->desc().shape_vec();
        shape_vec.erase(shape_vec.begin() + axis);
        auto stride_vec = input->desc().stride_vec();
        stride_vec.erase(stride_vec.begin() + axis);

        output->desc().set_shape_vec(shape_vec);
        output->desc().set_stride_vec(stride_vec);
        return {output};
    }
};

class OpUnsqueezeDesc : public OpDesc {
public:
    OpUnsqueezeDesc() : axis(0) {};
    OpUnsqueezeDesc(ssize_t axis) : axis(axis) {}
    virtual ~OpUnsqueezeDesc() = default;

    ssize_t axis;
};

class OpUnsqueeze : public Op {
public:
    NCG_OP_DEF_NAME(OpUnsqueeze);

    virtual void check_inputs(OpContext &ctx, const TensorVec &inputs) {
        NCG_OP_CHECK_NR_INPUTS(ctx, inputs, 1);

        const auto input = inputs[0];
        auto axis = this->template desc<OpUnsqueezeDesc>().axis;
        if (axis < 0) axis += input->desc().dim();

        NCG_OP_CHECK_INPUT_DIM_GEQ(ctx, inputs, 0, axis - 1);
    }

    virtual TensorVec compute(OpContext &ctx, const TensorVec &inputs) {
        const auto input = inputs[0];
        auto axis = this->template desc<OpUnsqueezeDesc>().axis;
        if (axis < 0) axis += input->desc().dim();
        auto output = tensor(input->desc(), input->storage(), false);

        auto shape_vec = input->desc().shape_vec();
        shape_vec.insert(shape_vec.begin() + axis, 1);
        auto stride_vec = input->desc().stride_vec();
        stride_vec.insert(stride_vec.begin() + axis, 0);

        output->desc().set_shape_vec(shape_vec);
        output->desc().set_stride_vec(stride_vec);
        return {output};
    }
};

class OpAutoBroadcast : public Op {
    NCG_OP_DEF_NAME(OpAutoBroadcast);

    virtual void check_inputs(OpContext &ctx, const TensorVec &inputs) {
        NCG_OP_CHECK_NONEMPTY_INPUTS(ctx, inputs);
        NCG_OP_CHECK_BROADCASTABLE_SHAPE(ctx, inputs);
    }

    virtual TensorVec compute(OpContext &ctx, const TensorVec &inputs) {
        auto shape = inputs[0]->desc().shape_vec();

        for (ssize_t i = 0; i < inputs.size(); ++i) {
            for (ssize_t j = 0; j < shape.size(); ++j) {
                shape[j] = std::max(inputs[i]->desc().shape(j), shape[j]);
            }
        }

        TensorVec outputs(inputs.size());
        auto expand_op = std::make_unique<OpExpand>();
        expand_op->set_desc(std::make_shared<OpExpandDesc>(shape));

        for (ssize_t i = 0; i < inputs.size(); ++i) {
            outputs[i] = expand_op->execute(ctx, {inputs[i]})[0];
        }

        return outputs;
    }
};

} /* !namespace ncg */

