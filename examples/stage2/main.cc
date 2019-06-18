/*
 * main.cc
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#include <cassert>
#include <cctype>
#include <cstring>
#include <cstdio>

#include <iostream>
#include <iomanip>
#include <unordered_map>
#include <string>
#include <vector>
#include <algorithm>

#include "ncg.h"

#include "custom_ops/print.h"
#include "custom_ops/assert.h"
#include "custom_ops/bind.h"

namespace {

using namespace ncg;

enum class COp : int {
    Assign = 0x0002,
    Cond = 0x0003,
    Print = 0X0004,
    Assert = 0x0005,
    Bind = 0x0006,
    Grad = 0x0007,
    GradAt = 0x0008,
    VarAssign = 0x0009,

    Sin = 0x1001,
    Log = 0x1002,
    Exp = 0x1003,
    Sigmoid = 0x1004,
    Tanh = 0x1005,

    Eq = 0x2001,
    Neq = 0x2002,
    Ge = 0x2003,
    Le = 0x2004,
    Geq = 0x2005,
    Leq = 0x2006,

    Add = 0x3001,
    Sub = 0x3002,
    Mul = 0x3003,
    Div = 0x3004,
};

inline bool operator < (COp lhs, COp rhs) {
    return static_cast<int>(lhs) < static_cast<int>(rhs);
}

class CValue {
public:
    CValue() : m_type('s'), v_string(), v_tensor() {}
    explicit CValue(const std::string &value) : m_type('s'), v_string(value), v_tensor() {}
    explicit CValue(const GTensorPtr &tensor) : m_type('t'), v_string(), v_tensor(tensor) {}
    virtual ~CValue() = default;

    char type() const { return m_type; }
    std::string get_string() const { return v_string; }
    GTensorPtr get_tensor() const { return v_tensor; }

private:
    char m_type;
    std::string v_string;
    GTensorPtr v_tensor;
};

class CParser {
public:
    CParser() : m_graph(), m_session(m_graph), m_variables(), m_op_stack(), m_value_stack() {}
    virtual ~CParser() = default;
    void parse_gheader(std::istream &);
    void parse_gdef(std::istream &);
    void parse_geval(std::istream &, std::vector<TensorPtr> &, bool verbose=false);

    GTensorPtr placeholder(const std::string &name) {
        return m_graph.op<GOpPlaceholder>(name, OpDescPtr(new GOpPlaceholderDesc(DTypeName::Float32, {})));
    }
    GTensorPtr constant(const std::string &name, float value) {
        auto tensor = scalar(DTypeName::Float32, value);
        return m_graph.op<GOpConstant>(name, OpDescPtr(new GOpConstantDesc(tensor)));
    }
    GTensorPtr variable(const std::string &name, float value) {
        auto tensor = scalar(DTypeName::Float32, value);
        return m_graph.op<GOpVariable>(name, OpDescPtr(new GOpVariableDesc(tensor)));
    }

    GTensorPtr arith_binary_op(COp op, const GTensorPtr &x, const GTensorPtr &y) {
        switch (op) {
            case COp::Add: return m_graph.op<GOpAdd>(nullptr, x, y);
            case COp::Sub: return m_graph.op<GOpSub>(nullptr, x, y);
            case COp::Mul: return m_graph.op<GOpMul>(nullptr, x, y);
            case COp::Div: return m_graph.op<GOpDiv>(nullptr, x, y);
            case COp::Ge: return m_graph.op<GOpGe>(nullptr, x, y);
            case COp::Le: return m_graph.op<GOpLe>(nullptr, x, y);
            case COp::Geq: return m_graph.op<GOpGeq>(nullptr, x, y);
            case COp::Leq: return m_graph.op<GOpLeq>(nullptr, x, y);
            case COp::Eq: return m_graph.op<GOpEq>(nullptr, x, y);
            case COp::Neq: return m_graph.op<GOpNeq>(nullptr, x, y);
            default: break;
        }
        return nullptr;
    }
    GTensorPtr arith_unary_op(COp op, const GTensorPtr &x) {
        switch (op) {
            case COp::Sin: return m_graph.op<GOpSin>(nullptr, x);
            case COp::Log: return m_graph.op<GOpLog>(nullptr, x);
            case COp::Exp: return m_graph.op<GOpExp>(nullptr, x);
            case COp::Sigmoid: return m_graph.op<GOpSigmoid>(nullptr, x);
            case COp::Tanh: return m_graph.op<GOpTanh>(nullptr, x);
            default: break;
        }
        return nullptr;
    }

    void debug() const {
        std::cerr << "==========Graph Def==========" << std::endl;
        for (auto &it : m_variables) {
            std::cerr << it.first << ": " << *(it.second) << std::endl;
        }
    }

private:
    void reduce_and_shift_op_(COp);
    void reduce_();
    void shift_value_(const CValue &);
    GTensorPtr pop_value_();
    std::string pop_string_();

    Graph m_graph;
    Session m_session;
    std::unordered_map<std::string, GTensorPtr> m_variables;
    std::vector<COp> m_op_stack;
    std::vector<CValue> m_value_stack;
};

void CParser::parse_gheader(std::istream &ss) {
    std::string name;
    char type;
    ss >> name >> type;
    if (type == 'P') {
        m_variables[name] = placeholder(name);
    } else if (type == 'V') {
        float value;
        ss >> value;
        m_variables[name] = variable(name, value);
    } else {
        ncg_assert(type == 'C');
        float value;
        ss >> value;
        m_variables[name] = constant(name, value);
    }
}

void CParser::parse_gdef(std::istream &ss) {
    std::string line;
    std::getline(ss, line);

    m_op_stack.clear();
    m_value_stack.clear();

    size_t pos = 0, last = 0;
    std::string token;

    while (true) {
        pos = line.find(" ", last);
        if (pos == std::string::npos) {
            token = line.substr(last);
        } else {
            token = line.substr(last, pos - last);
        }

#define PARSE_TOKEN(value, name) else if (token == #value) { reduce_and_shift_op_(COp::name); }

        if (token == "=") { reduce_and_shift_op_(COp::Assign); }
        PARSE_TOKEN(COND, Cond)
        PARSE_TOKEN(PRINT, Print)
        PARSE_TOKEN(ASSERT, Assert)
        PARSE_TOKEN(BIND, Bind)
        PARSE_TOKEN(GRAD, Grad)
        PARSE_TOKEN(AT, GradAt)
        PARSE_TOKEN(ASSIGN, VarAssign)
        PARSE_TOKEN(SIN, Sin)
        PARSE_TOKEN(LOG, Log)
        PARSE_TOKEN(EXP, Exp)
        PARSE_TOKEN(SIGMOID, Sigmoid)
        PARSE_TOKEN(TANH, Tanh)
        PARSE_TOKEN(+, Add)
        PARSE_TOKEN(-, Sub)
        PARSE_TOKEN(*, Mul)
        PARSE_TOKEN(/, Div)
        PARSE_TOKEN(==, Eq)
        PARSE_TOKEN(!=, Neq)
        PARSE_TOKEN(>, Ge)
        PARSE_TOKEN(<, Le)
        PARSE_TOKEN(>=, Geq)
        PARSE_TOKEN(<=, Leq)
        else { shift_value_(CValue(token)); }

#undef PARSE_TOKEN

        if (pos != std::string::npos) {
            last = pos + 1;
        } else {
            break;
        }
    }
    while (m_op_stack.size() > 1) {
        reduce_();
    }

    ncg_assert(m_op_stack[0] == COp::Assign);

    if (true) {
        const auto &name = m_value_stack[0].get_string();
        const auto &value = m_value_stack[1];
        GTensorPtr tensor;
        if (value.type() == 's') {
            tensor = m_variables[value.get_string()];
        } else {
            tensor = value.get_tensor();
        }

        m_variables[name] = tensor;
    }
}

void CParser::parse_geval(std::istream &ss, std::vector<TensorPtr> &answer_stack, bool verbose) {
    std::string op;
    ss >> op;
    if (op == "EVAL") {
        if (verbose)
            std::cerr << "=========Graph Eval==========" << std::endl;
        std::string name; int k;
        ss >> name;
        if (ss.eof()) {
            k = 0;
        } else {
            ss >> k;
        }

        GraphForwardContext ctx(m_session);
        std::string fname; float v;
        while (k--) {
            ss >> fname >> v;
            ctx.feed(fname, scalar(DTypeName::Float32, v));
        }
        auto outputs = ctx.eval({m_variables[name]});
        if (ctx.ok()) {
            std::cout << (outputs[0]->as<DTypeName::Float32>())->data_ptr()[0] << std::endl;
            answer_stack.push_back(outputs[0]);
        } else {
            std::cout << "ERROR: " << ctx.error_str() << std::endl;
            answer_stack.push_back(nullptr);
        }

        if (verbose) {
            std::cerr << "Verbose:" << std::endl;
            for (auto it: m_variables) {
                ctx.reset_error();
                outputs = ctx.eval({it.second});
                if (ctx.ok()) {
                    std::cout << it.first << ": " << (outputs[0]->as<DTypeName::Float32>())->data_ptr()[0] << std::endl;
                }
            }
        }
    } else if (op == "SETCONSTANT") {
        std::string name; float value;
        ss >> name >> value;
        auto variable_op =  m_variables[name]->owner_op<GOpVariable>();
        variable_op->set_value(m_session, scalar(DTypeName::Float32, value));

        answer_stack.emplace_back(scalar(DTypeName::Float32, 0));
    } else if (op == "SETANSWER") {
        std::string name; int k;
        ss >> name >> k;
        auto variable_op =  m_variables[name]->owner_op<GOpVariable>();
        auto value = answer_stack[k - 1];
        variable_op->set_value(m_session, value);

        answer_stack.emplace_back(scalar(DTypeName::Float32, 0));
    }
}

void CParser::reduce_and_shift_op_(COp op) {
    while (m_op_stack.size() > 0 && m_op_stack[m_op_stack.size() - 1] > op) {
        reduce_();
    }
    m_op_stack.emplace_back(op);
}
void CParser::shift_value_(const CValue &value) {
    m_value_stack.emplace_back(value);
}
GTensorPtr CParser::pop_value_() {
    auto v = *(m_value_stack.rbegin());
    m_value_stack.pop_back();
    if (v.type() == 's') {
        return m_variables[v.get_string()];
    }
    return v.get_tensor();
}
std::string CParser::pop_string_() {
    auto v = *(m_value_stack.rbegin());
    m_value_stack.pop_back();
    ncg_assert(v.type() == 's');
    return v.get_string();
}

void CParser::reduce_() {
    auto op = *(m_op_stack.rbegin());
    m_op_stack.pop_back();
    if (op == COp::Add || op == COp::Sub || op == COp::Mul || op == COp::Div) {
        auto b = pop_value_(), a = pop_value_();
        m_value_stack.emplace_back(CValue(arith_binary_op(op, a, b)));
    } else if (op == COp::Ge || op == COp::Le || op == COp::Geq || op == COp::Leq || op == COp::Eq || op == COp::Neq) {
        auto b = pop_value_(), a = pop_value_();
        m_value_stack.emplace_back(CValue(arith_binary_op(op, a, b)));
    } else if (op == COp::Sin || op == COp::Log || op == COp::Exp || op == COp::Sigmoid || op == COp::Tanh) {
        auto a = pop_value_();
        m_value_stack.emplace_back(CValue(arith_unary_op(op, a)));
    } else if (op == COp::Cond) {
        auto c = pop_value_(), b = pop_value_(), a = pop_value_();
        m_value_stack.emplace_back(CValue(m_graph.op<GOpCond>(nullptr, a, b, c)));
    } else if (op == COp::Print) {
        auto a = pop_string_();
        auto b = m_variables[a];
        m_value_stack.emplace_back(CValue(m_graph.op<GOpPrint>(OpDescPtr(new OpPrintDesc(a)), b)));
    } else if (op == COp::Assert) {
        auto a = pop_value_();
        m_value_stack.emplace_back(CValue(m_graph.op<GOpAssert>(nullptr, a)));
    } else if (op == COp::Bind) {
        auto b = pop_value_(), a = pop_value_();
        m_value_stack.emplace_back(CValue(m_graph.op<GOpBind>(nullptr, a, b)));
    } else if (op == COp::Grad) {
        auto a = pop_value_();
        m_graph.backward(a);
        m_value_stack.emplace_back(CValue(a));
    } else if (op == COp::GradAt) {
        auto b = pop_value_(), a = pop_value_();
        m_value_stack.emplace_back(CValue(b->grad(a)));
    } else if (op == COp::VarAssign) {
        auto b = pop_value_(), a = pop_value_();
        m_value_stack.emplace_back(CValue(m_graph.op<GOpAssign>(nullptr, a, b)));
    }
}

}

int main() {
    std::cout << std::fixed << std::setprecision(4);
    std::cerr << std::fixed << std::setprecision(4);

    auto parser = std::make_unique<CParser>();
    int n; std::string line;
    std::cin >> n; std::getline(std::cin, line);
    while (n--)
        parser->parse_gheader(std::cin);
    // parser->debug();
    std::cin >> n; std::getline(std::cin, line);
    while (n--)
        parser->parse_gdef(std::cin);
    // parser->debug();
    std::cin >> n; std::getline(std::cin, line);
    std::vector<ncg::TensorPtr> answer_stack;
    while (n--) {
        std::string temp;
        getline(std::cin, temp);
        auto ss = std::istringstream(temp);
        parser->parse_geval(ss, answer_stack);
    }

    return 0;
}

