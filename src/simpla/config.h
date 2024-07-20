#ifndef SIMPLA_CONFIG_H
#define SIMPLA_CONFIG_H

#define SIMPLA_VERSION_MAJOR 
#define SIMPLA_VERSION_MINOR 
#define SIMPLA_VERSION_PATCHLEVEL 
#define SIMPLA_VERSION_PATCHLEVEL 
#define SIMPLA_VERSION_IDENTIFY ""

#define AUTHOR ""
#define COPYRIGHT ""

/* Optimized build */
/* #undef OPT_BUILD */

/* Thread Building Blocks are available to use */
/* #undef TBB_FOUND */

/* BOOST headers are available to use */
/* #undef BOOST_FOUND */

/* BOOST headers are available to use */
/* #undef HAVE_BOOST_HEADERS */

/* #undef MPI_FOUND */

/* #undef SKIP_MPICXX */

/* #undef CUDA_FOUND */

/* HDF5 library is available so use it */
/* #undef HDF5_FOUND */

/* #undef LUA_FOUND */
/* BLAS library is available so use it */
/* #undef BLAS_FOUND */

/* #undef LAPACK_FOUND */

/* #undef PETSC_FOUND */

/* #undef SAMRAI_FOUND */

/* #undef SIMPLA_MAXIMUM_DIMENSION */

/* #undef SP_ARRAY_MAX_NDIMS */

#ifndef SP_ARRAY_MAX_NDIMS
#define SP_ARRAY_MAX_NDIMS 8
#endif

/* SLOW_FIRST: c-array  , FAST_FIRST: fortran-array slow-fist*/
/* #undef SP_ARRAY_DEFAULT_ORDER */

#ifndef SP_ARRAY_DEFAULT_ORDER
#define SP_ARRAY_DEFAULT_ORDER SLOW_FIRST
#endif
/* #undef SP_DEFAULT_SPACE_DIMS */
#ifndef SP_DEFAULT_SPACE_DIMS
#define SP_DEFAULT_SPACE_DIMS 3
#endif

/* SNaN: signaling nan,QNAN: quiet nan ,DENORM_MIN: 2^-1074*/
/* #undef SP_ARRAY_INITIALIZE_VALUE */

/* #undef SP_GEO_DEFAULT_TOLERANCE */
#ifndef SP_GEO_DEFAULT_TOLERANCE
#define SP_GEO_DEFAULT_TOLERANCE 1.0e-6
#endif

/* #undef SP_OUTPUT_SUFFIX */

#ifndef SP_OUTPUT_SUFFIX
#define SP_OUTPUT_SUFFIX SIMPLA
#endif

#include <stdint.h>
#include <stdlib.h>

/* #undef SP_REAL */
#ifndef SP_REAL
#define SP_REAL double
#endif

typedef SP_REAL Real;

#define SP_TRUE 1
#define SP_FALSE 0
#define SP_SUCCESS 0
#define SP_FAILED 1

#define SP_DO_NOTHING 0xFFFF

#define SP_UNIMPLEMENTED SP_DO_NOTHING + 1

typedef size_t size_type;
typedef int8_t byte_type;  // int8_t
typedef int Integral;
typedef int64_t index_type;
typedef unsigned int uint;
typedef size_type id_type;

#define SP_SUCCESS 0
#define SP_FAILED 1

#ifdef __cplusplus

#include <limits>

namespace simpla {
static constexpr Real SP_INFINITY = std::numeric_limits<Real>::infinity();
static constexpr Real SP_EPSILON = std::numeric_limits<Real>::epsilon();
static constexpr Real SP_SNaN = std::numeric_limits<Real>::signaling_NaN();
}
#endif  //__cplusplus

#endif  // SIMPLA_CONFIG_H
