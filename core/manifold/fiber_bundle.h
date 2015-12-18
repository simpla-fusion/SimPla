/**
 * @file fiber_bundle.h
 * @author salmon
 * @date 2015-10-18.
 */

#ifndef SIMPLA_FIBER_BUNDLE_H
#define SIMPLA_FIBER_BUNDLE_H

#include "../gtl/primitives.h"
#include "Properties.h"

namespace simpla { namespace manifold
{


template<typename ...> class FiberBundle;

template<typename P, typename M>
struct DirectMap
{
    typedef M mesh_type;
    typedef P point_type;
    typedef mesh_type::vector_type vector_type;

    DirectMap(mesh_type const &) { }

    ~DirectMap() { }

    inline typename mesh_type::point_type project(point_type const &p) const
    {
        return p.x;
    }

    inline point_type lift(typename mesh_type::point_type const &x, typename mesh_type::vector_type const &v) const
    {
        return point_type{x, v};
    }

    template<typename ...Others>
    void parallel_move(point_type *p, Real dt, vector_type const &a, Others &&... others) const
    {
        p->x += p->v * dt * 0.5;
        p->v += a * dt;
        p->x += p->v * dt * 0.5;

    }
};

/**
 * A fiber bundle is a structure (E, B, π, F), where E, B, and F
 * are topological spaces and \f$ \pi : E \mapto B \f$ is a continuous surjection
 * satisfying a local triviality condition outlined below. The space B is called
 * the '''Base space''' of the bundle, \f$E\f$ the total space, and \f$F\f$ the fiber.
 * The map \f$\pi\f$ is called the '''projection map''' (or '''bundle projection''').
 */
template<typename E, typename M, typename PI>
class FiberBundle : public PI
{
public:

    typedef E point_type; //!< coordinates in the total space,
    typedef M mesh_type; //!<  Base space;
    typedef PI project_map_type; //!<  projection map

    typedef Vec3 vector_type;
    typedef Real scalar_type;
private:
    typedef FiberBundle<point_type, mesh_type, project_map_type> this_type;
    typedef typename mesh_type::range_type range_type;
    typedef typename mesh_type::id_type id_type;


    project_map_type const &m_map_;

public:
    mesh_type const &m_mesh_;

    using project_map_type::project;
    using project_map_type::project_v;
    using project_map_type::lift;
    using project_map_type::parallel_move;
    using project_map_type::RBF;


    Properties properties;

    FiberBundle(mesh_type const &b) : m_map_(b), m_mesh_(b)
    {
    }

    FiberBundle(this_type const &other) : m_map_(other.m_map_), m_mesh_(other.m_mesh_)
    {
    }


    virtual  ~FiberBundle()
    {
    }

    void swap(this_type &other)
    {
        m_map_.swap(other);

        std::swap(m_mesh_, other.m_mesh_);
    }

    mesh_type const &mesh() const { return m_mesh_; }

    template<int IFORM>
    Real integral(mesh_type::point_type const &x0, point_type const &z, vector_type const &dx) const
    {
        return RBF((x - project(z)) / dx);
    }

    template<int IFORM>
    Real integral(id_type const &s, point_type const &z) const
    {
        auto x0 = m_mesh_.point(s);
        auto dx = m_mesh_.dx(s);

        return RBF((x - project(z)) / dx) * project_v<IFORM>(z, m_mesh_.sub_index(s));
    }
};

/**
 *
 *  `FiberBundle<P,M>` represents a fiber bundle \f$ \pi:P\to M\f$
 */



}}//namespace simpla{namespace manifold


#endif //SIMPLA_FIBER_BUNDLE_H
