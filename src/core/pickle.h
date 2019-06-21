/*
 * pickle.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#pragma once

#include "core/common.h"

#include <string>
#include <iostream>
#include <fstream>
#include <memory>
#include <utility>

namespace ncg {

enum class NCGPickleTypes : int32_t {
    Int64 = 0x0001,
    String = 0x0002,
    Int64Array = 0x0003,
    CharArray = 0x0004
};

#define WP(x) reinterpret_cast<const char *>(x)
#define RP(x) reinterpret_cast<char *>(x)

class NCGPickler {
public:
    NCGPickler(std::ostream &out) : m_out(out), m_own_stream(false) { ncg_assert(bool(m_out)); }
    NCGPickler(std::string fname) : m_own_out(fname, std::ios::out | std::ios::binary), m_own_stream(true), m_out(m_own_out) { ncg_assert(bool(m_out)); }

    NCGPickler(const NCGPickler &) = delete;
    NCGPickler(NCGPickler &&) = delete;

    void close() {
        if (m_own_stream) {
            m_own_out.close();
            m_own_stream = false;
        }
    }

    virtual ~NCGPickler() { close(); }

    void write(const int64_t &val) {
        int32_t type_val = static_cast<int32_t>(NCGPickleTypes::Int64);
        m_out.write(WP(&type_val), sizeof(type_val));
        m_out.write(WP(&val), sizeof(val));
    }

    void write(const std::string &val) {
        int32_t type_val = static_cast<int32_t>(NCGPickleTypes::String);
        m_out.write(WP(&type_val), sizeof(type_val));
        int64_t size_val = val.size();
        m_out.write(WP(&size_val), sizeof(size_val));
        const char *cstr_val = val.c_str();
        m_out.write(WP(cstr_val), sizeof(char) * (size_val + 1));
    }

    void write_ssize_array(const ssize_t *arr_val, int64_t size_val) {
        int32_t type_val = static_cast<int32_t>(NCGPickleTypes::Int64Array);
        m_out.write(WP(&type_val), sizeof(type_val));
        m_out.write(WP(&size_val), sizeof(size_val));

        for (ssize_t i = 0; i < size_val; ++i) {
            int64_t val = static_cast<int64_t>(arr_val[i]);
            m_out.write(WP(&val), sizeof(val));
        }
    }

    template <typename T = char>
    void write_char_array(const T *arr_val, int64_t size_val) {
        int32_t type_val = static_cast<int32_t>(NCGPickleTypes::CharArray);
        m_out.write(WP(&type_val), sizeof(type_val));
        m_out.write(WP(&size_val), sizeof(size_val));
        m_out.write(WP(arr_val), sizeof(T) * size_val);
    }

protected:
    std::ostream &m_out;

    std::ofstream m_own_out;
    bool m_own_stream;
};

class NCGUnpickler {
public:
    NCGUnpickler(std::istream &in) : m_in(in) { ncg_assert(bool(m_in)); }
    NCGUnpickler(std::string fname) : m_own_in(fname, std::ios::in | std::ios::binary), m_own_stream(true), m_in(m_own_in) { ncg_assert(bool(m_in)); }

    NCGUnpickler(const NCGUnpickler &) = delete;
    NCGUnpickler(NCGUnpickler &&) = delete;

    void close() {
        if (m_own_stream) {
            m_own_in.close();
            m_own_stream = false;
        }
    }

    virtual ~NCGUnpickler() { close(); }

    int64_t read_int64() {
        int32_t type_val = 0;
        m_in.read(RP(&type_val), sizeof(type_val));
        ncg_assert(type_val == static_cast<int32_t>(NCGPickleTypes::Int64));

        int64_t val = 0;
        m_in.read(RP(&val), sizeof(val));
        return val;
    }

    std::string read_string() {
        int32_t type_val = 0;
        m_in.read(RP(&type_val), sizeof(type_val));
        ncg_assert(type_val == static_cast<int32_t>(NCGPickleTypes::String));

        int64_t size_val = 0;
        m_in.read(RP(&size_val), sizeof(size_val));
        char *cstr_val = new char[size_val + 1];
        m_in.read(RP(cstr_val), sizeof(char) * (size_val + 1));
        std::string val(cstr_val, size_val);
        delete []cstr_val;

        return val;
    }

    std::pair<std::unique_ptr<ssize_t>, size_t> read_ssize_array() {
        int32_t type_val = 0;
        m_in.read(RP(&type_val), sizeof(type_val));
        ncg_assert(type_val == static_cast<int32_t>(NCGPickleTypes::Int64Array));

        int64_t size_val = 0;
        m_in.read(RP(&size_val), sizeof(size_val));
        int64_t *i64_val = new int64_t[size_val];
        m_in.read(RP(i64_val), sizeof(int64_t) * size_val);
        ssize_t *arr_val = new ssize_t[size_val];
        for (int64_t i = 0; i < size_val; ++i) {
            arr_val[i] = i64_val[i];
        }
        delete []i64_val;

        return std::make_pair(std::unique_ptr<ssize_t>(arr_val), static_cast<size_t>(size_val));
    }

    template <typename T = char>
    std::pair<std::unique_ptr<T>, size_t> read_char_array() {
        int32_t type_val = 0;
        m_in.read(RP(&type_val), sizeof(type_val));
        ncg_assert(type_val == static_cast<int32_t>(NCGPickleTypes::CharArray));

        int64_t size_val = 0;
        m_in.read(RP(&size_val), sizeof(size_val));
        char *arr_val = new char[size_val * sizeof(T)];
        m_in.read(RP(arr_val), sizeof(T) * size_val);

        return std::make_pair(std::unique_ptr<T>(reinterpret_cast<T *>(arr_val)), static_cast<size_t>(size_val));
    }

protected:
    std::istream &m_in;

    std::ifstream m_own_in;
    bool m_own_stream;
};

#undef RP
#undef WP


} /* !namespace ncg */

