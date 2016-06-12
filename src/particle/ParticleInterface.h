//
// Created by salmon on 16-6-8.
//

#ifndef SIMPLA_PARTICLECOMMON_H
#define SIMPLA_PARTICLECOMMON_H

#include <stddef.h>
#include "../sp_config.h"
#include "../data_model/DataTypeExt.h"


#ifdef __cplusplus
extern "C" {
#endif
#define POINT_HEAD long _tag;  Real r[3];

struct point_head
{
    POINT_HEAD
    char data[];
};
#define SP_DEFINE_PARTICLE(_S_NAME_, ...)   SP_DEFINE_C_STRUCT(_S_NAME_,long,_tag,Real[3], r, __VA_ARGS__)
#define SP_DEFINE_PARTICLE_DESCRIBE(_S_NAME_, ...)   SP_DEFINE_STRUCT_DESCRIBE(_S_NAME_,long,_tag, Real[3], r,__VA_ARGS__)

enum ParticleMomentType
{
    DENSITY,
    CURRENT,  //v
    KINETIC_ENERGY //0.5*m*(v*v)
};
#ifdef __cplusplus
}// extern "C" {
#endif

#endif //SIMPLA_PARTICLECOMMON_H
