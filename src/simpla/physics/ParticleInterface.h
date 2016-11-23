//
// Created by salmon on 16-6-8.
//

#ifndef SIMPLA_PARTICLECOMMON_H
#define SIMPLA_PARTICLECOMMON_H

#include <stddef.h>
#include "../sp_def.h"
#include "../toolbox/DataTypeExt.h"

#ifdef __cplusplus
extern "C"
{
#endif
/**
 *
 *  r    : local coordinate r\in [0,1]
 *
 *
 *               11             00              01
 * ---------+------------+-------@--------+-------------+---------------
 *r=       -1.5         -0.5     0       0.5           1.5
 *  particle only storage local relative topology_coordinate in cell  ,
 *  cell id is storage in the page
 */
//#define POINT_HEAD  SP_ENTITY_HEAD  Real r[3];
//
//struct point_head
//{
//	POINT_HEAD
//	byte_type *data_block;
//};

#define SP_DEFINE_PARTICLE(_S_NAME_, ...)   SP_DEFINE_C_STRUCT(_S_NAME_,uint64_t,_tag,Real[3], r, __VA_ARGS__)
#define SP_DEFINE_PARTICLE_DESCRIBE(_S_NAME_, ...)   SP_DEFINE_STRUCT_DESCRIBE(_S_NAME_,uint64_t,_tag, Real[3], r,__VA_ARGS__)

#ifdef __cplusplus
} // extern "C" {
#endif

#endif //SIMPLA_PARTICLECOMMON_H
