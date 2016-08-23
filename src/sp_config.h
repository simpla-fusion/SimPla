/**
 * sp_config.h
 *
 *    @date 2011-12-24
 *    @author  salmon
 */

#ifndef SIMPLA_DEFS_H_
#define SIMPLA_DEFS_H_


#include <stdint.h>
#include <stdlib.h>

#define  AUTHOR " YU Zhi <yuzhi@ipp.ac.cn> "
#define  COPYRIGHT "All rights reserved. (2016 )"


#define SP_TRUE 1

#define SP_FALSE  0

#define  SP_REAL double
//#define REAL_IS_FLOAT SP_FALSE

#define SP_DO_NOTHING 0xFFFF
#define SP_UNIMPLEMENT  SP_DO_NOTHING +1

#define SP_SUCCESS 1

#define SP_FAILED  0

typedef size_t size_type;

typedef int8_t byte_type; // int8_t

typedef SP_REAL Real;

typedef int Integral;

typedef int64_t id_type; //!< Data type of vertex's index , i.e. i,j

typedef int64_t index_type;

typedef union
{
    struct { int8_t w, z, y, x; };
    int32_t v;
} MeshEntityId32;

typedef union
{
    struct { int16_t w, z, y, x; };
    int64_t v;
} MeshEntityId64;


#endif /* SIMPLA_DEFS_H_ */
