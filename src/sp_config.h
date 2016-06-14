/**
 * sp_config.h
 *
 *    @date 2011-12-24
 *    @author  salmon
 */

#ifndef SIMPLA_DEFS_H_
#define SIMPLA_DEFS_H_

#include <stdint.h>

//#ifdef __cplusplus
//extern "C"
//{
//#endif

#define  AUTHOR " YU Zhi <yuzhi@ipp.ac.cn> "
#define  COPYRIGHT "All rights reserved. (2016 )"


typedef int8_t byte_type; // int8_t
typedef double Real;
typedef long Integral;
typedef int64_t id_type;
typedef int64_t index_type;
typedef uint64_t size_type;

//
//#ifdef __cplusplus
//};
//#endif


#ifdef __CUDACC__
#   define MC_DEVICE __device__
#   define MC_GLOBAL __global__
#   define MC_HOST   __host__
#   define MC_SHARED  __shared__
#   define MC_INLINE __device__ __forceinline__
#else
#   define MC_DEVICE
#   define MC_GLOBAL
#   define MC_HOST
#   define MC_SHARED
#   define MC_INLINE extern inline
#endif
#endif /* SIMPLA_DEFS_H_ */
