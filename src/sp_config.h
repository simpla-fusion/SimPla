/**
 * sp_config.h
 *
 *    @date 2011-12-24
 *    @author  salmon
 */

#ifndef SIMPLA_DEFS_H_
#define SIMPLA_DEFS_H_


#ifdef __cplusplus
extern "C"
{
#include <stdint.h>
#endif

#define  AUTHOR " YU Zhi <yuzhi@ipp.ac.cn> "
#define  COPYRIGHT "All rights reserved. (2016 )"


typedef int8_t ByteType; // int8_t
typedef double Real;
typedef long Integral;
typedef int64_t id_type;
typedef int64_t index_type;
typedef uint64_t size_type;


#ifdef __cplusplus
};
#endif
#endif /* SIMPLA_DEFS_H_ */
