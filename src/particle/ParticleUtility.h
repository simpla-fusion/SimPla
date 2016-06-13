//
// Created by salmon on 16-6-11.
//

#ifndef SIMPLA_PARTICLEINTERPOLATE_H
#define SIMPLA_PARTICLEINTERPOLATE_H

#include <math.h>
#include "ParticleInterface.h"
#include "BucketContainer.h"
#include "../sp_config.h"
//#include "../mesh/MeshIdHasher.h"
//
//#define CACHE_EXTENT_X 4
//#define CACHE_EXTENT_Y 4
//#define CACHE_EXTENT_Z 4
//#define CACHE_SIZE (CACHE_EXTENT_X*CACHE_EXTENT_Y*CACHE_EXTENT_Z)
//
//
//#define IX  1
//#define IY  3
//#define IZ  9
//
//
//#define ll -0.5
//#define rr 0.5
//
//extern inline void cache_gather(Real *v, Real const f[CACHE_SIZE], Real const *r0, const Real *r1)
//{
//    Real r[3] = {r0[0] - r1[0], r0[1] - r1[1], r0[2] - r1[2]};
//    id_type s = (int) (r[0]) * IX + (int) (r[1]) * IY + (int) (r[2]) * IZ;
//
//    *v = f[s + IX + IY + IZ /* */] * (r[0] - ll) * (r[1] - ll) * (r[2] - ll) +
//         f[s + IX + IY  /*     */] * (r[0] - ll) * (r[1] - ll) * (rr - r[2]) +
//         f[s + IX + IZ  /*     */] * (r[0] - ll) * (rr - r[1]) * (r[2] - ll) +
//         f[s + IX  /*          */] * (r[0] - ll) * (rr - r[1]) * (rr - r[2]) +
//         f[s + IY + IZ  /*     */] * (rr - r[0]) * (r[1] - ll) * (r[2] - ll) +
//         f[s + IY /*           */] * (rr - r[0]) * (r[1] - ll) * (rr - r[2]) +
//         f[s + IZ  /*          */] * (rr - r[0]) * (rr - r[1]) * (r[2] - ll) +
//         f[s  /*               */] * (rr - r[0]) * (rr - r[1]) * (rr - r[2]);
//}
//
//extern inline void cache_scatter(Real f[CACHE_SIZE], Real v, Real const *r0, Real const *r1)
//{
//    Real r[3] = {r0[0] - r1[0], r0[1] - r1[1], r0[2] - r1[2]};
//    id_type s = (int) (r[0]) * IX + (int) (r[1]) * IY + (int) (r[2]) * IZ;
//
//    f[s + IX + IY + IZ /*  */] += v * (r[0] - ll) * (r[1] - ll) * (r[2] - ll);
//    f[s + IX + IY /*       */] += v * (r[0] - ll) * (r[1] - ll) * (rr - r[2]);
//    f[s + IX + IZ /*       */] += v * (r[0] - ll) * (rr - r[1]) * (r[2] - ll);
//    f[s + IX /*            */] += v * (r[0] - ll) * (rr - r[1]) * (rr - r[2]);
//    f[s + IY + IZ /*       */] += v * (rr - r[0]) * (r[1] - ll) * (r[2] - ll);
//    f[s + IY /*            */] += v * (rr - r[0]) * (r[1] - ll) * (rr - r[2]);
//    f[s + IZ /*            */] += v * (rr - r[0]) * (rr - r[1]) * (r[2] - ll);
//    f[s/*                  */] += v * (rr - r[0]) * (rr - r[1]) * (rr - r[2]);
//
//
//}
//
//#undef ll
//#undef rr
//#undef IX
//#undef IY
//#undef IZ


/**
 *  p->r+=inc_r
 *  @return [0,26] shift of  local cell id
 */
extern inline void update_tag(struct point_head *p)
{
    index_type D[3] = {(index_type) (p->r[0]), (index_type) (p->r[1]), (index_type) (p->r[2])};

    int tag[4] = {0, 1, 0, 3};

    p->r[0] -= D[0];
    p->r[1] -= D[1];
    p->r[2] -= D[2];

    p->_tag = (uint64_t) ((2 - D[0]) | ((2 - D[1]) << 2) | ((2 - D[1]) << 4));

}

void move_points(struct spPage *pg, struct spPage **res, struct spPage *pool)
{
//    SP_PAGE_FOREACH(struct boris_point_s, p, &pg)
//    {
////        move_point_one((struct point_head *) (p), res, pool);
//    }
}

size_t spInsertParticle(struct spPage **p, size_t N, size_t size_in_byte, const byte_type *src,
                        struct spPagePool *pool);


#endif //SIMPLA_PARTICLEINTERPOLATE_H
