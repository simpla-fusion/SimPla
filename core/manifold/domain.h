/**
 * @file domain.h
 *
 *  Created on: 2015-4-19
 *      Author: salmon
 */

#ifndef CORE_MESH_DOMAIN_H_
#define CORE_MESH_DOMAIN_H_

#include <stddef.h>
#include <algorithm>
#include <cstdbool>
#include <functional>
#include <set>
#include <tuple>
#include <type_traits>

#include "../dataset/dataspace.h"
#include "../gtl/macro.h"
#include "../gtl/mpl.h"
#include "../gtl/ntuple.h"

#include "manifold_traits.h"
#include "domain_traits.h"

namespace simpla
{

template<typename ...> struct Field;
template<typename ...> struct Domain;


template<typename TM, int IFORM>
struct Domain<TM, std::integral_constant<int, IFORM> > : public TM::range_type
{
private:
    typedef Domain<TM, std::integral_constant<int, IFORM> > this_type;

    typedef typename TM::range_type range_type;

public:

    typedef TM mesh_type;

    typedef typename mesh_type::id_type id_type;

    typedef typename mesh_type::point_type point_type;

    static constexpr int ndims = mesh_type::ndims;

    static constexpr int iform = IFORM;

    mesh_type const *m_mesh_;
public:

    using range_type::empty;
    using range_type::size;

    Domain(mesh_type const &m)
            : range_type(m.template range<IFORM>()), m_mesh_(&m)
    {
    }


    Domain(this_type const &other)
            : range_type(other), m_mesh_(other.m_mesh_)
    {
    }

    Domain(this_type &&other) : range_type(other), m_mesh_(other.m_mesh_)
    {
    }


    virtual ~Domain() { }

    virtual void swap(this_type &other)
    {
        range_type::swap(other);
        std::swap(m_mesh_, other.m_mesh_);
    }

    this_type &operator=(this_type const &other)
    {
        this_type(other).swap(*this);
        return *this;
    }


    mesh_type const &mesh() const
    {
        return m_mesh_;
    }

public:

    template<typename TOP, typename ...Args>
    void action(TOP const &op, Args &&...args) const
    {
        for (auto const &s:*this)
        {
            op(m_mesh_->access(std::forward<Args>(args), s)...);
        }
    }

    void deploy() { }


    template<typename TI>
    bool in_box(TI const &idx) const
    {
        return range_type::in_box(idx);
    }

    bool in_box(point_type const &x) const
    {
        return range_type::in_box(mesh_type::coordinates_to_topology(x));
    }

    std::tuple<point_type, point_type> box() const
    {
        auto ext = range_type::box();

        return std::make_tuple(m_mesh_->point(std::get<0>(ext)),
                               m_mesh_->point(std::get<1>(ext)));
    }


    void reset(point_type const &b, point_type const &e)
    {
        range_type::reset(m_mesh_->coordinates_to_topology(b),
                          m_mesh_->coordinates_to_topology(e));
    }


};

} // namespace simpla

#endif /* CORE_MESH_DOMAIN_H_ */
