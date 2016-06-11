/**
 * @file ParticleEngine.h
 *
 * @date    2014-8-29  AM10:36:23
 * @author salmon
 */

#ifndef PARTICLE_ENGINE_H_
#define PARTICLE_ENGINE_H_

#include <stddef.h>
#include "../data_model/DataTypeExt.h"

#ifdef __cplusplus
extern "C" {
#endif
#define POINT_HEAD long _cell; long _tag;

struct point_head
{
    POINT_HEAD
    char data[];
};

#ifdef __cplusplus
}// extern "C" {
#endif

#define SP_DEFINE_PARTICLE(_S_NAME_, ...)   SP_DEFINE_C_STRUCT(_S_NAME_,long,_cell,long,_tag,__VA_ARGS__)
#define SP_DEFINE_PARTICLE_TYPE_ID(_S_NAME_, ...)   SP_DEFINE_C_STRUCT_TYPE_ID(_S_NAME_,long,_cell,long,_tag,__VA_ARGS__)


#endif /* PARTICLE_ENGINE_H_ */
