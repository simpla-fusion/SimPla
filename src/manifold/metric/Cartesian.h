/** 
 * @file Cartesian.h
 * @author salmon
 * @date 16-5-26 - 上午7:20
 *  */

#ifndef SIMPLA_CARTESIAN_H
#define SIMPLA_CARTESIAN_H

#include "../../mesh/MeshEntity.h"
#include "../../mesh/CoRectMesh.h"

namespace simpla { namespace manifold { namespace metric
{
template<typename> class Cartesian;

template<>
class Cartesian<mesh::CoRectMesh>
{
    typedef Cartesian<mesh::CoRectMesh> this_type;
    typedef mesh::CoRectMesh mesh_type;

    mesh_type &m_;

    typedef typename mesh::MeshEntityId id_type;
public:
    typedef this_type metric_policy;
    typedef Real scalar_type;

    Cartesian(mesh_type &m) : m_(m) { }

    ~Cartesian() { }

    static std::string class_name() { return "Cartesian"; }

    void deploy() {}

    std::ostream &print(std::ostream &os, int indent = 1) const
    {
        os << std::setw(indent) << " " << "Geometry={ Type=\"Cartesian\" }," << std::endl;
        return os;
    }


};

}}}//namespace simpla { namespace manifold{ namespace mertic

#endif //SIMPLA_CARTESIAN_H
