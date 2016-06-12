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


typedef int8_t ByteType; // int8_t
typedef double Real;
typedef long Integral;
typedef int64_t id_type;
typedef int64_t index_type;
typedef uint64_t size_type;

//
//#ifdef __cplusplus
//};
//#endif

//#ifdef USE_CUDA
//#   define CUDA_DEVICE __device__
//#   define CUDA_GLOBAL __global__
//#   define CUDA_HOST   __host__
//#else
//#   define CUDA_DEVICE
//#   define CUDA_GLOBAL
//#   define CUDA_HOST
//#endif
#endif /* SIMPLA_DEFS_H_ */
