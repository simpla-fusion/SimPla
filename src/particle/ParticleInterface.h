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
 *  _tag : relative cell shift  _tag= 0bzzyyxx
 *         _tag= 0bzzyyxx
 *               10,00-> 0, 01->1, 11->-1
 *  r    : local coordinate r\in [0,1]
 *
 *  cell index I[0] = page.cell_tag.x + (2-_tag&0b000011)%2 )
 *  coordinate x[0] = I[0] + r[0]*dx[0]
 *  particle only storage local relative coordinate in cell  ,
 *  cell id is storage in the page
 */
#define POINT_HEAD int _tag;  Real r[3];

struct point_head
{
    POINT_HEAD
    char data[];
};
struct page_head
{

    id_type cell_id;
    status_tag_type tag;
    struct spPage *next;

    size_type ele_size_in_byte;
    byte_type data[];

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
