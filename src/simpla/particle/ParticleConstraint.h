/**
 * @file ParticleConstraint.h
 *
 * @date    2014-9-2  AM10:39:25
 * @author salmon
 */

#ifndef PARTICLE_CONSTRAINT_H_
#define PARTICLE_CONSTRAINT_H_


#include "ParticleOld.h"

#include "../toolbox/Parallel.h"
#include "../../../to_delete/Constraint.h"


namespace simpla { namespace particle
{
template<typename P, typename M>
void reflect(ParticleOld<P, M> &p, geometry::Surface<M> const &surface)
{
    typedef typename geometry::Surface<M>::value_type constraint_value;
    typedef typename ParticleOld<P, M>::value_type particle_value;

    p.accept(surface.range(),
             [&](constraint_value const &item, particle_value &p)
             {
                 //fixme do sth.
             }
    );
};

template<typename P, typename M>
void absorb(ParticleOld<P, M> &p, geometry::Surface<M> const &surface)
{
    typedef typename geometry::Surface<M>::value_type constraint_value;
    typedef typename ParticleOld<P, M>::value_type particle_value;

    p.remove_if(surface.range(),
                [&](constraint_value const &item, particle_value const &p)
                {
                    //fixme do sth.
                    return false;
                }
    );

};

template<typename P, typename M, typename ...T, typename Func>
void transform(ParticleOld<P, M> &p, geometry::Constraint<M, T...> const &constraint, Func const &func)
{
    typedef typename geometry::Surface<M>::value_type constraint_value;
    typedef typename ParticleOld<P, M>::value_type particle_value;

    p.remove_if(constraint.range(),
                [&](constraint_value const &item, particle_value &p)
                {
                    //fixme do sth.
                }
    );

};


}}  // namespace simpla

#endif /* PARTICLE_CONSTRAINT_H_ */
