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
#include "mesh/mesh_common.h"
#include "manifold_traits.h"

namespace simpla
{


template<typename ...> struct BaseManifold;

template<typename TMetric, typename MeshType>
struct BaseManifold<TMetric, MeshType> : public MeshType, public TMetric
{
public:

    typedef TMetric metric_type;


    typedef MeshType mesh_type;

    typedef geometry::traits::coordinate_system_t<metric_type> coordinates_system_type;

    typedef geometry::traits::scalar_type_t<coordinates_system_type> scalar_type;

    typedef geometry::traits::point_type_t<coordinates_system_type> point_type;

    typedef geometry::traits::vector_type_t<coordinates_system_type> vector_type;

    using mesh_type::ndims;

private:

    typedef BaseManifold<metric_type, mesh_type> this_type;
public:
    BaseManifold() { }

    BaseManifold(this_type const &other) : mesh_type(other) { }

    virtual ~BaseManifold() { }

    template<typename TDict>
    void load(TDict const &dict) { mesh_type::load(dict); }

    template<typename OS>
    OS &print(OS &os) const
    {
        os << "\t Gemetry = {" << std::endl;

        os

        << "\t\tCoordinateSystem = {  Type = \""

        << traits::type_id<coordinates_system_type>::name() << "\",  }," << std::endl;

        mesh_type::print(os);

        os << "\t }, #BaseManifold " << std::endl;

        return os;
    }

    virtual void deploy()
    {
        mesh_type::deploy();
        mesh_type::update_volume(*this);
    };
    using mesh_type::volume;
    using mesh_type::dual_volume;
    using mesh_type::inv_volume;
    using mesh_type::inv_dual_volume;

    using metric_type::inner_product;

}; //struct BaseManifold<CS,TopologyTags >



}//namespace simpla
#endif //SIMPLA_BASE_MANIFOLD_H
