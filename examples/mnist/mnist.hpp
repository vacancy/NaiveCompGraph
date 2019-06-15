/*
 * mnist.hpp
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

namespace mnist {

int32_t reverse_int(int32_t i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return (int32_t(ch1) << 24) + (int32_t(ch2) << 16) + (int32_t(ch3) << 8) + ch4;
}

int32_t read_int(std::istream &file) {
    int32_t int_value = 0;
    file.read(reinterpret_cast<char *>(&int_value), sizeof(int_value));
    return reverse_int(int_value);
}

std::vector<std::vector<float>> read_mnist_image(std::string filename) {
    std::ifstream file(filename, std::ios::binary);
    std::vector<std::vector<float>> output;
    if (file.is_open()) {
        int32_t magic_number = read_int(file);
        int32_t nr_images = read_int(file);
        int32_t nr_rows = read_int(file);
        int32_t nr_cols = read_int(file);

        output.resize(nr_images, std::vector<float>(nr_rows * nr_cols));

        for (int i = 0; i < nr_images; ++i) {
            for (int r = 0; r < nr_rows; ++r) {
                for (int c = 0; c < nr_cols; ++c) {
                    unsigned char temp = 0;
                    file.read(reinterpret_cast<char *>(&temp), sizeof(temp));
                    output[i][(nr_rows * r) + c] = (static_cast<float>(temp) / 255) - 0.5;
                }
            }
        }
    }

    return output;
}

std::vector<int32_t> read_mnist_label(std::string filename) {
    std::ifstream file(filename, std::ios::binary);
    std::vector<int32_t> output;
    if (file.is_open()) {
        int32_t magic_number = read_int(file);
        int32_t nr_images = read_int(file);
        output.resize(nr_images);

        for (int i = 0; i < nr_images; ++i) {
            unsigned char temp = 0;
            file.read(reinterpret_cast<char *>(&temp), sizeof(temp));
            output[i] = int32_t(temp);
        }
    }

    return output;
}

void print_data(std::vector<std::vector<float>> images, std::vector<int32_t> labels, ssize_t index) {
    using namespace std;

    cerr << "Image #" << index << " (Label: " << labels[index] << ")" << endl;
    for (ssize_t i = 0; i < 28; ++i) {
        for (ssize_t j = 0; j < 28; ++j) {
            if (j != 0) cerr << " ";
            cerr << ((images[index][i * 28 + j] > 0) ? 'X' : ' ');
        }
        cerr << endl;
    }
}

} /* !namespace mnist */

