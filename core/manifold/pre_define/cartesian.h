/**
 * @file cartesian_corectmesh.h
 * @author salmon
 * @date 2015-10-28.
 */

#ifndef SIMPLA_CARTESIAN_CORECTMESH_H
#define SIMPLA_CARTESIAN_CORECTMESH_H

#include "predefine.h"
#include "../../geometry/cs_cartesian.h"
#include "../topology/topology.h"
#include "../topology/corect_mesh.h"

namespace simpla
{
namespace manifold
{
template<int NDIMS, typename TopologyType=topology::CoRectMesh>
using Cartesian=DefaultManifold<geometry::Metric<::simpla::geometry::coordinate_system::Cartesian<NDIMS>>, TopologyType>;
}//namespace manifold
//
//
//template<typename ...> struct BaseManifold;
//template<typename ...> struct Metric;
//
//template<int NDIMS, typename TopologyType>
//struct BaseManifold<Metric<coordinate_system::Cartesian<NDIMS>>, TopologyType>
//        : public TopologyType, public Metric<coordinate_system::Cartesian<NDIMS>>
//{
//public:
//    typedef coordinate_system::Cartesian<NDIMS, 2> coordinates_system_type;
//
//    typedef TopologyType topology_type;
//
//    typedef Metric<coordinate_system::Cartesian<NDIMS>> metric_type;
//
//    typedef traits::scalar_type_t<coordinates_system_type> scalar_type;
//
//    typedef traits::point_type_t<coordinates_system_type> point_type;
//
//    typedef traits::vector_type_t<coordinates_system_type> vector_type;
//
//    using topology_type::ndims;
//
//private:
//    typedef BaseManifold<coordinates_system_type, topology_type> this_type;
//public:
//    BaseManifold() { }
//
//    BaseManifold(this_type const &other) : topology_type(other) { }
//
//    virtual BaseManifoldfold() { }
//
//
//    template<typename OS>
//    OS &print(OS &os) const
//    {
//        os << "\t Gemetry = {" << std::endl;
//
//        os
//
//        << "\t\tCoordinateSystem = {  Type = \""
//
//        << traits::type_id<coordinates_system_type>::name() << "\",  }," << std::endl;
//
//        topology_type::print(os);
//
//        os << "\t }, #BaseManifold " << std::endl;
//
//        return os;
//    }
//
//    virtual void deploy()
//    {
//        topology_type::deploy();
//        topology_type::update_volume((*this));
//    }
//
//    using topology_type::load;
//
//    using topology_type::volume;
//    using topology_type::dual_volume;
//    using topology_type::inv_volume;
//    using topology_type::inv_dual_volume;
//
//    using metric_type::inner_product;
//
////    template<typename T0, typename T1, typename ...Args>
////    constexpr auto inner_product(T0 const &v0, T1 const &v1, Args &&...args) const
////    DECL_RET_TYPE((simpla::inner_product(v0, v1)))
//};

}//namespace simpla
#endif //SIMPLA_CARTESIAN_CORECTMESH_H
