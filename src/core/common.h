/*
 * common.h
 * Copyright (C) 2019
 *
 * Distributed under terms of the MIT license.
 */

#ifndef CORE_COMMON_H
#define CORE_COMMON_H

#include <cstdlib>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <typeinfo>

namespace ncg {

typedef unsigned int uint;
typedef long long int64;
typedef unsigned long long uint64;

inline bool is_a_gt_zero_and_lt_b(int a, int b) {
    return static_cast<uint>(a) < static_cast<uint>(b);
}

// Begin OpenMP {{

#ifdef USE_OPENMP
    #include <omp.h>
    #if defined(_WIN32) || defined(__WIN32__)
        #define ompfor __pragma(omp parallel for) for
        #define omplock __pragma(omp critical)
    #else
        #define ompfor _Pragma("omp parallel for") for
        #define omplock _Pragma("omp critical")
    #endif
    const int kNumThreads = omp_get_max_threads();
    inline int omp_thread_id() { return omp_get_thread_num(); }
#else  // USE_OPENMP
    #define ompfor for
    #define omplock
    const int kNumThreads = 1;
    inline int omp_thread_id() { return 0; }
#endif  // USE_OPENMP

// End OpenMP }}

// Begin Alignment {{

#if defined(_WIN32) || defined(__WIN32__)
    #define align_attrib(typ, siz) __declspec(align(siz)) typ
#else
    #define align_attrib(typ, siz) typ __attribute__((aligned(siz)))
#endif

#if defined(_WIN32) || defined(__WIN32__)
inline void* align_alloc(size_t size, size_t alignsize) {
    return _aligned_malloc(size, alignsize);
}
inline void align_free(void* mem) { _aligned_free(mem); }
#else
inline void* align_alloc(size_t size, size_t alignsize) {
    void* mem = nullptr;
    int ret = posix_memalign((void**)&mem, alignsize, size);
    return (ret == 0) ? mem : nullptr;
}
inline void align_free(void* mem) { free(mem); }
#endif

// End Alignment }}

// Begin Arithmetic {{

#if defined(_WIN32) || defined(__WIN32__)
    #if _MSC_VER <= 1600
        #define isnan(x) _isnan(x)
        #define isinf(x) (!_finite(x))
    #endif
#endif

// End Arithmetic }}

// Begin Assertion {{

#ifndef __FUNCTION_NAME__
    #if defined(_WIN32) || defined(__WIN32__)
        #define __FUNCTION_NAME__ __FUNCTION__
    #else
        #define __FUNCTION_NAME__ __func__
    #endif
#endif

#define ncg_assert_msg(PREDICATE, MSG) \
do { \
    if (!(PREDICATE)) { \
        std::cerr << "Asssertion \"" \
        << #PREDICATE << "\" failed in " << __FILE__ \
        << " line " << __LINE__ \
        << " in function \"" << (__FUNCTION_NAME__) << "\"" \
        << " : " << (MSG) << std::endl; \
        std::abort(); \
    } \
} while (false)

#define ncg_assert(PREDICATE) \
do { \
    if (!(PREDICATE)) { \
        std::cerr << "Asssertion \"" \
        << #PREDICATE << "\" failed in " << __FILE__ \
        << " line " << __LINE__ \
        << " in function \"" << (__FUNCTION_NAME__) << "\"" << std::endl; \
        std::abort(); \
    } \
} while (false)

#ifndef NDEBUG
#define ncg_dassert_msg(PREDICATE, MSG) ncg_assert_msg(PREDICATE, MSG)
#define ncg_dassert(PREDICATE) ncg_assert(PREDICATE)
#else  // NDEBUG
#define ncg_dassert_msg(PREDICATE, MSG) do {} while(false)
#define ncg_dassert(PREDICATE) do {} while (false)
#endif // NDEBUG

// End Assertion }}

} /* !namespace ncg */

#endif /* !CORE_COMMON_H */
