/**
 * @file mesh.h
 * @author salmon
 * @date 2015-10-13.
 */

#ifndef SIMPLA_GEOMETRY_H
#define SIMPLA_GEOMETRY_H

#include "../../gtl/primitives.h"
#include "../../gtl/macro.h"
#include "../../gtl/type_traits.h"
#include "../../geometry/coordinate_system.h"

#include "../topology/topology_common.h"
#include "geometry_traits.h"

namespace simpla
{


template<typename ...> struct Geometry;


template<typename CS, typename TopologyType>
struct Geometry<CS, TopologyType> : public TopologyType
{
public:
    typedef CS coordinates_system_type;

    typedef TopologyType topology_type;

    typedef traits::scalar_type_t<coordinates_system_type> scalar_type;

    typedef traits::point_type_t<coordinates_system_type> point_type;

    typedef traits::vector_type_t<coordinates_system_type> vector_type;

    using topology_type::ndims;

private:

    typedef Geometry<CS, TopologyType> this_type;

    mertic<coordinates_system_type> m_metric_;

public:
    Geometry() { }

    Geometry(this_type const &other) : topology_type(other) { }

    virtual ~Geometry() { }


    template<typename TDict>
    void load(TDict const &dict) { topology_type::load(dict); }

    template<typename OS>
    OS &print(OS &os) const
    {
        os << "\t Gemetry = {" << std::endl;

        os

        << "\t\tCoordinateSystem = {  Type = \"" << traits::type_id<CS>::name() << "\",  }," << std::endl;

        topology_type::print(os);

        os << "\t }, #Geometry " << std::endl;

        return os;
    }

    virtual void deploy() { topology_type::deploy(); };



/** @} */
/** @name Volume
 * @{
 */

private:

    Real volume_(id_type s) const
    {
        return m_metric_.volume(topology_type::node_id(s), topology_type::point(s));
    }

    Real dual_volume_(id_type s) const
    {
        return m_metric_.dual_volume(topology_type::node_id(s), topology_type::point(s));
    }

public:

    Real volume(id_type s) const
    {
        return topology_type::volume(s) * volume_(s);
    }

    Real dual_volume(id_type s) const
    {
        return topology_type::dual_volume(s) * dual_volume_(s);
    }

    Real inv_volume(id_type s) const
    {
        return topology_type::inv_volume(s) / volume_(s);
    }

    Real inv_dual_volume(id_type s) const
    {
        return topology_type::inv_dual_volume(s) / dual_volume_(s);
    }

/**@}*/


    template<typename T0, typename T1, typename ...Args>
    auto inner_product(T0 const &v0, T1 const &v1, Args &&...args) const
    DECL_RET_TYPE((m_metric_.inner_product(v0, v1, std::forward<Args>(args)...)))

}; //struct Geometry<CS,TopologyTags >



}//namespace simpla
#endif //SIMPLA_GEOMETRY_H
