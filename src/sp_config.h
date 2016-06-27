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

#ifndef USE_DOUBLE
#   define  SP_REAL float
#else
#   define  SP_REAL double
#endif

//
//#ifdef __cplusplus
//};
//#endif


#endif /* SIMPLA_DEFS_H_ */
