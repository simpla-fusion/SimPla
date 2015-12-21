/**
 * @file ParticleTracker.h
 * @author salmon
 * @date 2015-12-20.
 */

#ifndef SIMPLA_PARTICLETRACKER_H
#define SIMPLA_PARTICLETRACKER_H

#include "../data_model/DataTypeExt.h"
#include "../gtl/primitives.h"

namespace simpla
{
namespace particle
{

template<typename P>
class enable_tracking : public P
{
    typedef typename P::point_type value_type;

    SP_DEFINE_STRUCT(point_type, size_t, _tag, value_type, p);


    template<typename ...Args>
    enable_tracking(Args &&...args) : P(std::forward<Args>(args)...) { }

    ~enable_tracking() { }

    Vec3 project(point_type const &p) const { return P::project(p.p); }

    std::tuple<Vec3, Vec3> push_forward(point_type const &p) const
    {
        return P::push_forward(p.p);
    }

    template<typename ...Args>
    point_type lift(Args &&...args) const
    {
        return point_type{0, P::lift(std::forward<Args>(args)...)};
    }

    template<typename ...Args>
    point_type sample(Args &&...args) const
    {
        return point_type{0, P::lift(std::forward<Args>(args)...)};
    }

    template<typename ...Args>
    void integral(Vec3 const &x, point_type const &p, Args &&...args) const
    {
        P::integral(x, p.p, std::forward<Args>(args)...);
    }


    template<typename ...Args>
    void push(point_type *p, Args &&...args) const
    {
        P::push(*(p->p), std::forward<Args>(args)...);

    };

};

template<typename ...> class Tracker;

template<typename P>
class Tracker<P> : public P::mesh_type::AttributeEntity
{
    typedef typename P::mesh_type mesh_type;
    typedef P::mesh_type::AttributeEntity base_type;

    Tracker(mesh_type const &m, std::string const &s_name) : base_type(m, s_name) { }

    virtual ~Tracker() { }

    data_model::DataSet data_set() const
    {

    }
};
}
}//namespace simpla { namespace particle{

#endif //SIMPLA_PARTICLETRACKER_H
