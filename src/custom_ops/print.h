/*
 * print.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#ifndef CUSTOM_OPS_PRINT_H
#define CUSTOM_OPS_PRINT_H

#include "ncg.h"
#include <string>

namespace ncg {

struct OpPrintDesc : OpDesc {
public:
    OpPrintDesc(const std::string &name = "") : name(name) {}
    std::string name;
};

class GOpPrint : public GraphOp, public GraphSingleOutputOp {
public:
    NCG_GOP_DEF_NAME(GOpPrint);

    virtual void check_inputs(Graph &graph, const GTensorVec &inputs) {
        NCG_OP_CHECK_NR_INPUTS(graph, inputs, 1);
    }
    virtual GTensorVec init_outputs(Graph &graph, const GTensorVec &inputs) {
        return {make_tensor(0, inputs[0]->desc())};
    }

    virtual void forward(GraphForwardContext &ctx) const {
        TensorPtr tensor = ctx.tensor(m_inputs[0]);
        const std::string &name = this->template desc<OpPrintDesc>().name;
        std::cout << "PRINT operator: " << name << " = " << tensor->as<DTypeName::Float32>()->data_ptr()[0] << std::endl;
        ctx.set_tensor(m_outputs[0], tensor);
    }

    virtual void backward(Graph &graph, GTensorPtr loss) {
        auto output_grad = m_outputs[0]->grad(loss);
        m_inputs[0]->set_grad(graph, loss, output_grad);
    }
};

} /* !namespace ncg */

#endif /* !CUSTOM_OPS_PRINT_H */
