/**
 * @file particle_constraint.h
 *
 * @date    2014-9-2  AM10:39:25
 * @author salmon
 */

#ifndef PARTICLE_CONSTRAINT_H_
#define PARTICLE_CONSTRAINT_H_


#include "particle.h"

#include "Parallel.h"
#include "../model/constraint.h"


namespace simpla { namespace particle
{
template<typename P, typename M>
void reflect(Particle<P, M> &p, model::Surface<M> const &surface)
{
    typedef typename model::Surface<M>::value_type constraint_value;
    typedef typename Particle<P, M>::value_type particle_value;

    p.accept(surface.range(),
             [&](constraint_value const &item, particle_value &p)
             {
                 //fixme do sth.
             }
    );
};

template<typename P, typename M>
void absorb(Particle<P, M> &p, model::Surface<M> const &surface)
{
    typedef typename model::Surface<M>::value_type constraint_value;
    typedef typename Particle<P, M>::value_type particle_value;

    p.remove_if(surface.range(),
                [&](constraint_value const &item, particle_value const &p)
                {
                    //fixme do sth.
                    return false;
                }
    );

};

template<typename P, typename M, typename ...T, typename Func>
void transform(Particle<P, M> &p, model::Constraint<M, T...> const &constraint, Func const &func)
{
    typedef typename model::Surface<M>::value_type constraint_value;
    typedef typename Particle<P, M>::value_type particle_value;

    p.remove_if(constraint.range(),
                [&](constraint_value const &item, particle_value &p)
                {
                    //fixme do sth.
                }
    );

};


}}  // namespace simpla

#endif /* PARTICLE_CONSTRAINT_H_ */
