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


template<typename TMetric, typename TopologyType>
struct Geometry<TMetric, TopologyType> : public TopologyType, public TMetric
{
public:

    typedef TMetric metric_type;


    typedef TopologyType topology_type;

    typedef traits::coordinate_system_t<metric_type> coordinates_system_type;

    typedef traits::scalar_type_t<coordinates_system_type> scalar_type;

    typedef traits::point_type_t<coordinates_system_type> point_type;

    typedef traits::vector_type_t<coordinates_system_type> vector_type;

    using topology_type::ndims;

private:

    typedef Geometry<metric_type, topology_type> this_type;
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

        << "\t\tCoordinateSystem = {  Type = \""

        << traits::type_id<coordinates_system_type>::name() << "\",  }," << std::endl;

        topology_type::print(os);

        os << "\t }, #Geometry " << std::endl;

        return os;
    }

    virtual void deploy()
    {
        topology_type::deploy();
        topology_type::update_volume(*this);
    };
    using topology_type::volume;
    using topology_type::dual_volume;
    using topology_type::inv_volume;
    using topology_type::inv_dual_volume;

    using metric_type::inner_product;

}; //struct Geometry<CS,TopologyTags >



}//namespace simpla
#endif //SIMPLA_GEOMETRY_H
