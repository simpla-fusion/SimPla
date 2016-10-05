/**
 * @file ParticleTracker.h
 * @author salmon
 * @date 2015-12-20.
 */

#ifndef SIMPLA_PARTICLETRACKER_H
#define SIMPLA_PARTICLETRACKER_H

#include "../toolbox/DataTypeExt.h"
#include "../sp_def.h"

namespace simpla { namespace particle
{

template<typename P>
class enable_tracking : public P
{

    typedef typename P::sample_type value_type;
public:
    typedef typename P::point_type point_type;
    typedef typename P::vector_type vector_type;


    SP_DEFINE_STRUCT(sample_type, size_t, _tag, value_type, p);


    template<typename ...Args>
    enable_tracking(Args &&...args) : P(std::forward<Args>(args)...) { }

    ~enable_tracking() { }

    point_type project(sample_type const &p) const { return P::project(p.p); }

    std::tuple<point_type, vector_type> push_forward(point_type const &p) const
    {
        return P::push_forward(p.p);
    }

    template<typename ...Args>
    sample_type lift(Args &&...args) const
    {
        return sample_type{0, P::lift(std::forward<Args>(args)...)};
    }

    template<typename ...Args>
    sample_type sample(Args &&...args) const
    {
        return sample_type{0, P::lift(std::forward<Args>(args)...)};
    }

    template<typename ...Args>
    void integral(point_type const &x, sample_type const &p, Args &&...args) const
    {
    }


    template<typename ...Args>
    void push(sample_type *p, Args &&...args) const
    {
        P::push(&(p->p), std::forward<Args>(args)...);

    };

};


}}//namespace simpla { namespace particle{

#endif //SIMPLA_PARTICLETRACKER_H
