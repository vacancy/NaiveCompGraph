/*
 * shape.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#ifndef CORE_OPS_SHAPE_H
#define CORE_OPS_SHAPE_H

#include "core/op.h"
#include "core/ops/op_common.h"
#include <algorithm>
#include <memory>
#include <vector>

namespace ncg {

class OpReshapeDesc : public OpDesc {
public:
    OpReshapeDesc() : shape() {}
    OpReshapeDesc(const shape_vec &shape) : shape(shape) {}
    virtual ~OpReshapeDesc() = default;

    shape_vec shape;
};

class OpReshape : public Op {
public:
    virtual void check_inputs(OpContext &ctx, const TensorVec &inputs) {
        if (inputs.size() == 1) {
            ctx.error(this) << "Reshape op accept only 1 input.";
        }

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
                ctx.error(this) << "Invalid shape " << shape;
            }
        }

        if (nr_negative == 1) {
            if (inputs[0]->desc().numel() % nr_total != 0) {
                ctx.error(this) << "Invalid shape " << shape;
            }
        } else if (nr_negative == 0) {
            if (inputs[0]->desc().numel() != nr_total) {
                ctx.error(this) << "Invalid shape " << shape;
            }
        } else {
            ctx.error(this) << "Invalid shape " << shape;
        }
    }

    virtual TensorVec compute(OpContext &ctx, const TensorVec &inputs) {
        auto shape = this->template desc<OpReshapeDesc>().shape;
        auto input = inputs[0];

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

        input->make_contiguous();
        TensorDesc desc(input->desc().dtype(), shape);
        TensorPtr output = tensor(desc, input->storage(), false);

        return {output};
    }
};

class OpPermuteDesc : public OpDesc {
public:
    OpPermuteDesc() : axes() {}
    OpPermuteDesc(const shape_vec &axes) : axes(axes) {}
    virtual ~OpPermuteDesc() = default;

    shape_vec axes;
};

class OpPermute: public Op {
public:
    virtual void check_inputs(OpContext &ctx, const TensorVec &inputs) {
        if (inputs.size() == 1) {
            ctx.error(this) << "Permute op accept only 1 input.";
        }

        auto &axes = this->template desc<OpPermuteDesc>().axes;
        std::vector<bool> used(axes.size());
        for (ssize_t i = 0; i < axes.size(); ++i) {
            int j = axes[i];

            if (j >= 0 && j < axes.size() && !used[j]) {
                used[j] = true;
            } else {
                ctx.error(this) << "Invalid permute axes " << axes;
            }
        }
    }

    virtual TensorVec compute(OpContext &ctx, const TensorVec &inputs) {
        auto input = inputs[0];
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
    OpExpandDesc(const shape_vec &shape) : shape(shape) {}
    virtual ~OpExpandDesc() = default;

    shape_vec shape;
};

class OpExpand : public Op {
public:
    virtual void check_inputs(OpContext &ctx, const TensorVec &inputs) {
        if (inputs.size() == 1) {
            ctx.error(this) << "Expand op accept only 1 input.";
        }

        auto input = inputs[0];
        auto &shape = this->template desc<OpExpandDesc>().shape;

        if (shape.size() != input->desc().dim()) {
            ctx.error(this) << "Expand op cannot change the dimension.";
        }
        for (ssize_t i = 0; i < shape.size(); ++i) {
            if (shape[i] != input->desc().shape(i) && input->desc().shape(i) != 1) {
                ctx.error(this) << "Invalid expand op, from " << input->desc().shape_vec() << " to " << shape;
                break;
            }
        }
    }

    virtual TensorVec compute(OpContext &ctx, const TensorVec &inputs) {
        auto input = inputs[0];
        auto &shape = this->template desc<OpExpandDesc>().shape;
        auto output = tensor(input->desc(), input->storage(), false);

        for (ssize_t i = 0; i < input->desc().dim(); ++i) {
            if (shape[i] != input->desc().shape(i)) {
                output->desc().shape(i) = shape[i];
                output->desc().stride(i) = 0;
            }
        }
    }
};

class OpAutoBroadcast : public Op {
    virtual void check_inputs(OpContext &ctx, const TensorVec &inputs) {
        if (inputs.size() == 0) {
            ctx.error(this) << "AutoBroadcast op accept at least one input.";
        }

        for (ssize_t i = 0; i < inputs.size(); ++i) {
            if (inputs[i]->desc().dim() != inputs[0]->desc().dim()) {
                ctx.error(this) << "AutoBroadcast op support same-dimension inputs.";
            }
        }
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
            outputs[i] = expand_op->execute(ctx, inputs[i])[0];
        }

        return outputs;
    }
};

} /* !namespace ncg */

#endif /* !CORE_OPS_SHAPE_H */
