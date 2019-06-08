/*
 * slice.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#ifndef SLICE_H
#define SLICE_H

namespace ncg {

class OpConcatDesc : public OpDesc {
public:
    OpConcatDesc() : axis(0) {}
    OpConcatDesc(ssize_t axis) : axis(axis) {}
    virtual ~OpConcatDesc() = default;

    ssize_t axis;
};

class OpConcat : public Op {
    virtual void check_inputs(OpContext &ctx, const TensorVec &inputs) {
        if (inputs.size() == 0) {
            ctx.error(this) << "Concat op accept at least one input.";
        }

        auto axis = this->template desc<OpConcatDesc>().axis;

        for (ssize_t i = 0; i < inputs.size(); ++i) {
            if (inputs[i]->desc().dim() != inputs[0]->desc().dim()) {
                ctx.error(this) << "Concat op support same-dimension inputs.";
            }
            for (ssize_t j = 0; j < inputs[0]->desc().dim(); ++j) {
                if (j != axis) {
                    if (inputs[i]->desc().shape(j) != inputs[0]->desc().shape(j)) {
                        ctx.error(this) << "Concat op: inputs shape can only differ along the " << axis << " dimension.";
                    }
                }
            }
        }
    }

    virtual TensorVec execute(OpContext &ctx, const TensorVec &inputs) {
        auto axis = this->template desc<OpConcatDesc>().axis;

        auto shape = inputs[0]->desc().shape_vec();
        for (ssize_t i = 1; i < inputs.size(); ++i) {
            shape[axis] += inputs[i]->desc().shape(axis);
        }

        auto output = empty(inputs[0]->desc().dtype(), shape);
        ssize_t index = 0;

        for (ssize_t i = 0; i < inputs.size(); ++i) {
            for (ssize_t j = 0; j < inputs[i]->desc().numel(); ++j) {Q
                ssize_t k = output->elindex(j) + index * output->desc().stride(axis);
                output->mutable_data_ptr()[k] = inputs[i]->elat(j);
            }

            index += inputs->[i]->desc().shape(axis);
        }
    }
};

class OpSplitDesc : public OpDesc {
public:
    OpSplitDesc() : axis(0), splits() {}
    OpSplitDesc(ssize_t axis, const shape_vec &splits) : axis(axis), splits(splits) {}
    virtual ~OpSplitDesc() = default;

    ssize_t axis;
    shape_vec splits;
};

class OpSplit : public Op {
public:

    virtual void check_inputs(OpContext &ctx, const TensorVec &inputs) {
        if (inputs.size() != 1) {
            ctx.error(this) << "Split op accepts only one input.";
        }

        auto &splits = this->template desc<OpSplitDesc>().splits;
        auto axis = this->template desc<OpSplitDesc>().axis;

        ssize_t nr_total = 0;
        for (ssize_t i = 0; i < splits.size(); ++i) {
            nr_total += splits[i];
        }

        if (nr_total != inputs[0]->desc().shape(axis)) {
            ctx.error(this) << "Split values are not consistent with the shape.";
        }
    }

    virtual TensorVec execute(OpContext &ctx, const TensorVec &inputs) {
        auto input = inputs[0];
        auto &splits = this->template desc<OpSplitDesc>().splits;
        auto axis = this->template desc<OpSplitDesc>().axis;
        TensorVec outputs(splits.size());

        ssize_t index = 0;
        for (ssize_t i = 0; i < splits.size(); ++i) {
            TensorDesc desc(input->desc().dtype(), input->desc().shape(), input->desc().stride());
            desc.shape(axis) = splits[i];
            ssize_t offset = index * desc.stride(axis);

            outputs[i] = tensor(desc, input->storage(), false, offset);
        }

        return outputs;
    }
};

} /* !namespace ncg */

#endif /* !SLICE_H */
