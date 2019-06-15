/*
 * main.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include "mnist.hpp"
#include "ncg.h"

namespace ncg {

std::ostream &print_matrix(std::ostream &out, ncg::TensorPtr tensor) {
    ncg_assert(tensor->desc().dim() == 2);
    out << "[";
    for (ssize_t i = 0; i < tensor->desc().shape(0); ++i) {
        if (i != 0) out << std::endl << " ";
        out << "[";
        for (ssize_t j = 0; j < tensor->desc().shape(1); ++j) {
            if (j != 0) out << ", ";
            out << tensor->as<ncg::DTypeName::Float32>()->at(i, j);
        }
        out << "],";
    }

    return out;
}

} /* !namespace ncg */

namespace mnist {

ncg::TensorPtr ncg_read_mnist_image(std::string filename) {
    auto images = ncg::fromcc(ncg::DTypeName::Float32, mnist::read_mnist_image(filename));

    auto ctx = ncg::OpContext();
    auto reshape_op = ncg::OpReshape();
    reshape_op.set_desc(ncg::OpDescPtr(new ncg::OpReshapeDesc({images->desc().shape(0), 1, 28, 28})));
    auto output_vec = reshape_op.execute(ctx, {images});
    ncg_assert_msg(!ctx.is_error(), ctx.error_str());
    return output_vec[0];
}

ncg::TensorPtr ncg_read_mnist_label(std::string filename) {
    return ncg::fromcc(ncg::DTypeName::Int32, mnist::read_mnist_label(filename));
}

void print_data(ncg::TensorPtr raw_images, ncg::TensorPtr raw_labels, ssize_t index) {
    auto images = raw_images->as<ncg::DTypeName::Float32>();
    auto labels = raw_labels->as<ncg::DTypeName::Int32>();

    std::cerr << "Image #" << index << " (Label: " << labels->at(index) << ")" << std::endl;
    for (ssize_t i = 0; i < 28; ++i) {
        for (ssize_t j = 0; j < 28; ++j) {
            if (j != 0) std::cerr << " ";
            std::cerr << ((images->at(index, ssize_t(0), i, j) > 0) ? 'X' : ' ');
        }
        std::cerr << std::endl;
    }
}

} /* !namespace mnist */

int main() {
    // auto train_images = mnist::ncg_read_mnist_image("./data/train-images-idx3-ubyte");
    // auto train_labels = mnist::ncg_read_mnist_label("./data/train-labels-idx1-ubyte");
    auto test_images = mnist::ncg_read_mnist_image("./data/t10k-images-idx3-ubyte");
    auto test_labels = mnist::ncg_read_mnist_label("./data/t10k-labels-idx1-ubyte");

    for (ssize_t i = 0; i < 10; ++i) {
        mnist::print_data(test_images, test_labels, i);
    }

    return 0;
}
