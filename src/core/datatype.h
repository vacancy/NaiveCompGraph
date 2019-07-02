/*
 * datatype.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#pragma once

#include "core/common.h"

namespace ncg {

/* Enum list of all supported dtypes. */
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
    /* typename cctype: the corresponding type in c++. */
    /* static constexpr char []name: human readable name as a string. */
    /* static constexpr DTypeName identifier: a DTypeName-typed identifier. */
};

template <typename T>
struct CCType {
    /* typename cctype: the corresponding type in c++. */
    /* static constexpr char []name: human readable name as a string. */
    /* static constexpr DTypeName identifier: a DTypeName-typed identifier. */
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

/* Helper macro: runs another macro with a single argument dtype_name, inside a switch block. */
#define NCG_DTYPE_SWITCH(dtype_name, MACRO_) case DTypeName::dtype_name: MACRO_(dtype_name); break;

/* Helper macro: runs another macro with a single argument dtype_name by switch-casing the value of an variable. */
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

/* Helper macro: runs another macro with a single argument dtype_name by switch-casing the value of an variable. */
/* The macro is invoked only when the input dtype is a float type (Float32 or Float64). */
#define NCG_DTYPE_SWITCH_FLOAT(dtype_var, MACRO) switch(dtype_var) { \
    NCG_DTYPE_SWITCH(Float32, MACRO); \
    NCG_DTYPE_SWITCH(Float64, MACRO); \
    default: break; \
}

/* Helper macro: dtype-ed function template instantiation. The input macro takes the dtype name as input and construct the full signature. */
#define NCG_INSTANTIATE_DTYPE(dtype_name, MACRO_) template MACRO_(dtype_name)
#define NCG_DTYPE_INSTANTIATE_ALL(MACRO) \
    NCG_INSTANTIATE_DTYPE(Int8, MACRO); \
    NCG_INSTANTIATE_DTYPE(UInt8, MACRO); \
    NCG_INSTANTIATE_DTYPE(Int32, MACRO); \
    NCG_INSTANTIATE_DTYPE(UInt32, MACRO); \
    NCG_INSTANTIATE_DTYPE(Int64, MACRO); \
    NCG_INSTANTIATE_DTYPE(UInt64, MACRO); \
    NCG_INSTANTIATE_DTYPE(Float32, MACRO); \
    NCG_INSTANTIATE_DTYPE(Float64, MACRO)

/* Helper macro: dtype-ed class template instantiation. */
#define NCG_DTYPE_INSTANTIATE_CLASS(dtype_name, class_name) template class class_name<DTypeName::dtype_name>
#define NCG_DTYPE_INSTANTIATE_CLASS_ALL(class_name) \
    NCG_DTYPE_INSTANTIATE_CLASS(Int8, class_name); \
    NCG_DTYPE_INSTANTIATE_CLASS(UInt8, class_name); \
    NCG_DTYPE_INSTANTIATE_CLASS(Int32, class_name); \
    NCG_DTYPE_INSTANTIATE_CLASS(UInt32, class_name); \
    NCG_DTYPE_INSTANTIATE_CLASS(Int64, class_name); \
    NCG_DTYPE_INSTANTIATE_CLASS(UInt64, class_name); \
    NCG_DTYPE_INSTANTIATE_CLASS(Float32, class_name); \
    NCG_DTYPE_INSTANTIATE_CLASS(Float64, class_name)

/* Return the human readable string, as the name of the given dtype. */
inline const char *get_dtype_name(DTypeName dtype) {
#define GET_NAME_DTYPE_CASE(dtype_name) return #dtype_name;
NCG_DTYPE_SWITCH_ALL(dtype, GET_NAME_DTYPE_CASE);
#undef GET_NAME_DTYPE_CASE
}


} /* !namespace ncg */

