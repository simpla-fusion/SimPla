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
/**
 *  uint64_t _tag;
 *   0b0000000000000000000000zzyyxx
 *   first 38 bits are vonstant index number of particle,
 *   last  6 bits 'zzyyxx' are relative cell shift
 *         000000 means particle is the correct cell
 *         if xx = 00 means particle is the correct cell
 *                 01 -> (+1) right neighbour cell
 *                 11 -> (-1)left neighbour cell
 *                 10  not neighbour, if xx=10 , r[0]>2 or r[0]<-1
 *
 *        |   001010   |    001000     |   001001      |
 * -------+------------+---------------+---------------+---------------
 *        |            |               |               |
 *        |            |               |               |
 * 000011 |   000010   |    000000     |   000001      |   000011
 *        |            |               |               |
 *        |            |               |               |
 * -------+------------+---------------+---------------+---------------
 *        |   000110   |    000100     |   000101      |
 *
 *  r    : local coordinate r\in [0,1]
 *
 *
 *               11             00              01
 * ---------+------------+-------@--------+-------------+---------------
 *r=       -1.5         -0.5     0       0.5           1.5
 *  particle only storage local relative coordinate in cell  ,
 *  cell id is storage in the page
 */
#define POINT_HEAD uint64_t _tag;  Real r[3];

struct point_head
{
    POINT_HEAD
    byte_type *data;
};

#define SP_DEFINE_PARTICLE(_S_NAME_, ...)   SP_DEFINE_C_STRUCT(_S_NAME_,uint64_t,_tag,Real[3], r, __VA_ARGS__)
#define SP_DEFINE_PARTICLE_DESCRIBE(_S_NAME_, ...)   SP_DEFINE_STRUCT_DESCRIBE(_S_NAME_,uint64_t,_tag, Real[3], r,__VA_ARGS__)


#ifdef __cplusplus
}// extern "C" {
#endif

#endif //SIMPLA_PARTICLECOMMON_H
