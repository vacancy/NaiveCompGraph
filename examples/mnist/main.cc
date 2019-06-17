/*
 * main.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include "mnist.hpp"
#include "ncg.h"
#include "nn/ops.h"

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

    ncg::OpContext ctx;
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

class DataLoader {
public:
    DataLoader(const ncg::TensorVec &data, ssize_t batch_size, std::mt19937 &rng) : m_data(data), m_batch_size(batch_size), m_shuffle_indices(), m_index(0), m_rng(rng) {}

    ncg::TensorVec next() {
        if (m_shuffle_indices == nullptr) {
            m_shuffle_indices = ncg::rand_permutation(m_rng, m_data[0]->desc().shape(0));
        }

        ssize_t begin = m_index, end = std::min(m_index + m_batch_size, m_data[0]->desc().shape(0));
        m_index += m_batch_size;

        ncg::TensorVec outputs;
        outputs.push_back(m_shuffle_indices.narrow(0, begin, end - begin));
        for (const auto &i : m_data) {
            outputs.push_back(i.index_select(0, m_shuffle_indices.narrow(0, begin, end - begin)));
        }

        if (m_index >= m_data[0]->desc().shape(0)) {
            m_shuffle_indices = nullptr;
            m_index = 0;
        }

        return outputs;
    }

    ssize_t epoch_size() const {
        ssize_t n = m_data[0]->desc().shape(0);
        return n / m_batch_size + (n % m_batch_size != 0);
    }

private:
    ncg::TensorVec m_data;
    ssize_t m_batch_size;

    ncg::TensorPtr m_shuffle_indices;
    ssize_t m_index;

    std::mt19937 &m_rng;
};

} /* !namespace mnist */

namespace mnist_model {

using namespace ncg;

struct MLPModel {
    MLPModel(std::mt19937 &rng) : rng(rng) {
        image = G::placeholder("image", {100, 784}, DTypeName::Float32);
        label = G::placeholder("label", {100}, DTypeName::Int64);
        linear1 = G::linear("linear1", image, 512, rng);
        activation1 = G::tanh(linear1);
        logits = G::linear("linear2", activation1, 10, rng);
        pred = logits.max(-1)[1];

        prob = G::softmax(logits, -1);
        loss = G::xent_sparse(prob, label, -1).mean(0);
        accuracy = (pred.eq(label)).float32().mean(0);
    }

    GTensorVec train_ops(float lr=0.01) {
        GTensorVec ops;
        auto &graph = get_default_graph();

        graph.backward(loss);
        for (const auto &name : {"linear1:W", "linear2:W", "linear1:b", "linear2:b"}) {
            auto W = graph.find_op(name)->outputs()[0];
            auto G = W->grad(loss);
            auto new_W = W - G * lr;
            ops.push_back(G::assign(W, new_W));
        }

        return ops;
    }

    std::mt19937 &rng;
    GTensorPtr image, label, linear1, activation1, logits, prob, pred, loss, accuracy;
};

} /* !namespace mnist_model */

int main() {
    std::random_device rd{};
    std::mt19937 rng{rd()};

    std::cerr << "Loading training data..." << std::endl;
    auto train_images = mnist::ncg_read_mnist_image("./data/train-images-idx3-ubyte");
    auto train_labels = mnist::ncg_read_mnist_label("./data/train-labels-idx1-ubyte");
    std::cerr << "Loading test data..." << std::endl;
    auto test_images = mnist::ncg_read_mnist_image("./data/t10k-images-idx3-ubyte");
    auto test_labels = mnist::ncg_read_mnist_label("./data/t10k-labels-idx1-ubyte");

    // for (ssize_t i = 0; i < 10; ++i) {
    //     mnist::print_data(test_images, test_labels, i);
    // }

    std::cerr << "Building data loaders..." << std::endl;
    auto train_loader = std::unique_ptr<mnist::DataLoader>(new mnist::DataLoader({train_images, train_labels}, 100, rng));
    auto test_loader = std::unique_ptr<mnist::DataLoader>(new mnist::DataLoader({test_images, test_labels}, 100, rng));

    std::cerr << "Building the MLP model..." << std::endl;
    auto model = std::make_unique<mnist_model::MLPModel>(rng);
    auto train_ops = model->train_ops();
    train_ops.insert(train_ops.begin() + 0, model->loss);
    train_ops.insert(train_ops.begin() + 1, model->accuracy);
    auto test_ops = {model->loss, model->accuracy};

    for (int i = 1; i <= train_loader->epoch_size() * 40; ++i) {
        auto inputs = train_loader->next();
        ncg::GraphForwardContext ctx;
        ctx.feed("image", inputs[1].reshape({-1, 784}));
        ctx.feed("label", inputs[2].cast(ncg::DTypeName::Int64));
        auto outputs = ctx.eval(train_ops);
        ncg_assert_msg(ctx.ok(), ctx.error_str());

        std::cerr << "Iteration [" << i / train_loader->epoch_size() + 1 << "::" << (i - 1) % train_loader->epoch_size() + 1 << "/" << train_loader->epoch_size() << "]: "
            << "loss = " << ncg::tocc_scalar<double>(outputs[0]) << ", "
            << "accuracy = " << ncg::tocc_scalar<double>(outputs[1]) << ".";
        if (i % 100 == 0) std::cerr << std::endl; else std::cerr << "\r";

        if (i % train_loader->epoch_size() == 0) {
            double loss = 0, accuracy = 0;
            ssize_t tot = 0;

            for (int j = 1; j <= test_loader->epoch_size(); ++j) {
                auto inputs = test_loader->next();
                ncg::GraphForwardContext ctx;
                ctx.feed("image", inputs[1].reshape({-1, 784}));
                ctx.feed("label", inputs[2].cast(ncg::DTypeName::Int64));
                auto outputs = ctx.eval(test_ops);
                ncg_assert_msg(ctx.ok(), ctx.error_str());

                std::cerr << "Evaluation [" << i << "::" << j << "/" << test_loader->epoch_size() << "]: "
                    << "loss = " << ncg::tocc_scalar<double>(outputs[0]) << ", "
                    << "accuracy = " << ncg::tocc_scalar<double>(outputs[1]) << ".";
                if (j % 100 == 0) std::cerr << std::endl; else std::cerr << "\r";

                loss += ncg::tocc_scalar<double>(outputs[0]);
                accuracy += ncg::tocc_scalar<double>(outputs[1]);
                tot += inputs[1]->desc().shape(0);
            }

            std::cerr << "Evaluation [" << i << "]: "
                << "loss = " << loss / tot << ", "
                << "accuracy = " << accuracy / tot << "."
                << std::endl;
        }
    }

    return 0;
}

