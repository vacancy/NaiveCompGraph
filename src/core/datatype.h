/*
 * datatype.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#pragma once

#include "core/common.h"

namespace ncg {

enum class DTypeName : int {
    Int8,
    UInt8,
    Int32,
    UInt32,
    Int64,
    UInt64,
    Float32,
    Float64,
};

template <DTypeName Name>
struct DType {
};

template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
struct CCType {
};

#define DEF_DTYPE_CCTYPE(identifier_, cctype_) template<> \
struct DType<DTypeName::identifier_> { \
    using cctype = cctype_; \
    static constexpr char name[] = #identifier_; \
    static constexpr DTypeName identifier = DTypeName::identifier_; \
}; \
template<> \
struct CCType<cctype_> { \
    using cctype = cctype_; \
    static constexpr char name[] = #identifier_; \
    static constexpr DTypeName identifier = DTypeName::identifier_; \
}

DEF_DTYPE_CCTYPE(Int8, int8_t);
DEF_DTYPE_CCTYPE(UInt8, uint8_t);
DEF_DTYPE_CCTYPE(Int32, int32_t);
DEF_DTYPE_CCTYPE(UInt32, uint32_t);
DEF_DTYPE_CCTYPE(Int64, int64_t);
DEF_DTYPE_CCTYPE(UInt64, uint64_t);
DEF_DTYPE_CCTYPE(Float32, float);
DEF_DTYPE_CCTYPE(Float64, double);

#define NCG_DTYPE_SWITCH(dtype_, MACRO_) case DTypeName::dtype_: MACRO_(dtype_); break;

#define NCG_DTYPE_SWITCH_ALL(dtype_var, MACRO) switch(dtype_var) { \
    NCG_DTYPE_SWITCH(Int8, MACRO); \
    NCG_DTYPE_SWITCH(UInt8, MACRO); \
    NCG_DTYPE_SWITCH(Int32, MACRO); \
    NCG_DTYPE_SWITCH(UInt32, MACRO); \
    NCG_DTYPE_SWITCH(Int64, MACRO); \
    NCG_DTYPE_SWITCH(UInt64, MACRO); \
    NCG_DTYPE_SWITCH(Float32, MACRO); \
    NCG_DTYPE_SWITCH(Float64, MACRO); \
}

#define NCG_DTYPE_SWITCH_FLOAT(dtype_var, MACRO) switch(dtype_var) { \
    NCG_DTYPE_SWITCH(Float32, MACRO); \
    NCG_DTYPE_SWITCH(Float64, MACRO); \
    default: break; \
}

#define NCG_INSTANTIATE_DTYPE(dtype_, MACRO_) template MACRO_(dtype_)

#define NCG_DTYPE_INSTANTIATE_ALL(MACRO) \
    NCG_INSTANTIATE_DTYPE(Int8, MACRO); \
    NCG_INSTANTIATE_DTYPE(UInt8, MACRO); \
    NCG_INSTANTIATE_DTYPE(Int32, MACRO); \
    NCG_INSTANTIATE_DTYPE(UInt32, MACRO); \
    NCG_INSTANTIATE_DTYPE(Int64, MACRO); \
    NCG_INSTANTIATE_DTYPE(UInt64, MACRO); \
    NCG_INSTANTIATE_DTYPE(Float32, MACRO); \
    NCG_INSTANTIATE_DTYPE(Float64, MACRO)

#define NCG_DTYPE_INSTANTIATE_CLASS(dtype_, class_name) template class class_name<DTypeName::dtype_>

#define NCG_DTYPE_INSTANTIATE_CLASS_ALL(class_name) \
    NCG_DTYPE_INSTANTIATE_CLASS(Int8, class_name); \
    NCG_DTYPE_INSTANTIATE_CLASS(UInt8, class_name); \
    NCG_DTYPE_INSTANTIATE_CLASS(Int32, class_name); \
    NCG_DTYPE_INSTANTIATE_CLASS(UInt32, class_name); \
    NCG_DTYPE_INSTANTIATE_CLASS(Int64, class_name); \
    NCG_DTYPE_INSTANTIATE_CLASS(UInt64, class_name); \
    NCG_DTYPE_INSTANTIATE_CLASS(Float32, class_name); \
    NCG_DTYPE_INSTANTIATE_CLASS(Float64, class_name)

inline const char *get_dtype_name(DTypeName dtype) {
#define GET_NAME_DTYPE_CASE(dtype_name) return #dtype_name;
NCG_DTYPE_SWITCH_ALL(dtype, GET_NAME_DTYPE_CASE);
#undef GET_NAME_DTYPE_CASE
}


} /* !namespace ncg */

