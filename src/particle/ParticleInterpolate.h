//
// Created by salmon on 16-6-11.
//

#ifndef SIMPLA_PARTICLEINTERPOLATE_H
#define SIMPLA_PARTICLEINTERPOLATE_H

#include "ParticleEngine.h"

double spGetValue(hash, f, s)

double gather(long s, double r0, double r1, double r2, doube *f, struct hash_s const &hash)
{
    return hash(f, ((s + X) + Y) + Z) * r0 * r1 * r2 //
           + hash(f, (s + X) + Y) * r0 * r1 * (1.0 - r2) //
           + hash(f, (s + X) + Z) * r0 * (1.0 - r1) * r2 //
           + hash(f, (s + X)) * r0 * (1.0 - r1) * (1.0 - r2) //
           + hash(f, (s + Y) + Z) * (1.0 - r[0]) * r1 * r2 //
           + hash(f, (s + Y)) * (1.0 - r0) * r1 * (1.0 - r2) //
           + hash(f, s + Z) * (1.0 - r0) * (1.0 - r1) * r2 //
           + hash(f, s) * (1.0 - r0) * (1.0 - r1) * (1.0 - r2);
}

double scatter(struct hash_s const *hash, long s, double *f, double d0, double d1, double d2, double r0, double r1,
               double r2)
{
    return spGetValue(hash, f, s) * std::abs((d0 - r0) * (d1 - r1) * (d2 - r2));


}

#endif //SIMPLA_PARTICLEINTERPOLATE_H
