/**
 * @file base_manifold.h
 * @author salmon
 * @date 2015-10-13.
 */

#ifndef SIMPLA_BASE_MANIFOLD_H
#define SIMPLA_BASE_MANIFOLD_H

#include "../gtl/primitives.h"
#include "../gtl/macro.h"
#include "../gtl/type_traits.h"
#include "../geometry/coordinate_system.h"
#include "topology/topology_common.h"
#include "manifold_traits.h"

namespace simpla
{


template<typename ...> struct BaseManifold;

template<typename TMetric, typename TopologyType>
struct BaseManifold<TMetric, TopologyType> : public TopologyType, public TMetric
{
public:

    typedef TMetric metric_type;


    typedef TopologyType topology_type;

    typedef geometry::traits::coordinate_system_t<metric_type> coordinates_system_type;

    typedef geometry::traits::scalar_type_t<coordinates_system_type> scalar_type;

    typedef geometry::traits::point_type_t<coordinates_system_type> point_type;

    typedef geometry::traits::vector_type_t<coordinates_system_type> vector_type;

    using topology_type::ndims;

private:

    typedef BaseManifold<metric_type, topology_type> this_type;
public:
    BaseManifold() { }

    BaseManifold(this_type const &other) : topology_type(other) { }

    virtual ~BaseManifold() { }

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

        os << "\t }, #BaseManifold " << std::endl;

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

}; //struct BaseManifold<CS,TopologyTags >



}//namespace simpla
#endif //SIMPLA_BASE_MANIFOLD_H
