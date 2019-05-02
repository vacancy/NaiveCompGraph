/*
 * print_op.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#ifndef PRINT_OP_H
#define PRINT_OP_H

#include "graph/graph_op.h"
#include <string>

namespace ncg {

struct GOpPrintDesc : OpDesc {
public:
    GOpPrintDesc(const std::string &name = "") : name(name) {}
    std::string name;
};

class GOpPrint : public GraphOp, public GraphSingleOutputOp {
public:
    NCG_DEF_GOPNAME(GOpPrint);

    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        if (inputs.size() != 1) {
            graph.error(this) << "Print op takes one input";
        }
    }
    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        return {make_tensor(0, inputs[0]->desc())};
    }

    virtual void forward(GraphForwardContext &ctx) const {
        TensorPtr tensor = ctx.tensor(m_inputs[0]);
        const std::string &name = this->template desc<GOpPrintDesc>().name;
        std::cout << "PRINT operator: " << name << " = " << tensor->as<DTypeName::Float32>()->data_ptr()[0] << std::endl;
        ctx.set_tensor(m_outputs[0], tensor);
    }
};

} /* !namespace ncg */

#endif /* !PRINT_OP_H */
