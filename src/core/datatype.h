/*
 * datatype.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#ifndef DATATYPE_H
#define DATATYPE_H

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

#define DEF_DTYPE_CCTYPE(identifier, cctype_) template<> \
struct DType<DTypeName::identifier> { \
    using cctype = cctype_; \
    static constexpr char name[] = #identifier; \
    static constexpr DTypeName identifier = DTypeName::identifier; \
}

DEF_DTYPE_CCTYPE(Int8, short);
DEF_DTYPE_CCTYPE(UInt8, unsigned short);
DEF_DTYPE_CCTYPE(Int32, long);
DEF_DTYPE_CCTYPE(UInt32, unsigned long);
DEF_DTYPE_CCTYPE(Int64, long long);
DEF_DTYPE_CCTYPE(UInt64, unsigned long long);
DEF_DTYPE_CCTYPE(Float32, float);
DEF_DTYPE_CCTYPE(Float64, double);

#define NCG_SWITCH_DTYPE(dtype_, MACRO_) case DTypeName::dtype_: MACRO_(dtype_); break;

#define NCG_SWITCH_DTYPE_ALL(dtype_var, MACRO) switch(dtype_var) { \
    NCG_SWITCH_DTYPE(Int8, MACRO); \
    NCG_SWITCH_DTYPE(UInt8, MACRO); \
    NCG_SWITCH_DTYPE(Int32, MACRO); \
    NCG_SWITCH_DTYPE(UInt32, MACRO); \
    NCG_SWITCH_DTYPE(Int64, MACRO); \
    NCG_SWITCH_DTYPE(UInt64, MACRO); \
    NCG_SWITCH_DTYPE(Float32, MACRO); \
    NCG_SWITCH_DTYPE(Float64, MACRO); \
}


} /* !namespace ncg */

#endif /* !DATATYPE_H */