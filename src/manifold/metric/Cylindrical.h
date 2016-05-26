/** 
 * @file Cylindrical.h
 * @author salmon
 * @date 16-5-26 - 上午7:24
 *  */

#ifndef SIMPLA_CYLINDRICAL_H
#define SIMPLA_CYLINDRICAL_H

#include "../../mesh/MeshEntity.h"
#include "../../mesh/CoRectMesh.h"

namespace simpla { namespace manifold { namespace metric
{

template<typename> class Cylindrical;

template<>
class Cylindrical<mesh::CoRectMesh>
{
    typedef Cylindrical<mesh::CoRectMesh> this_type;
    typedef mesh::CoRectMesh mesh_type;
    typedef typename mesh::MeshEntityId id_type;

    mesh_type const &m_;
public:
    typedef this_type metric_policy;

    Cylindrical(mesh_type const &m) : m_(m) { }

    ~Cylindrical() { }

public:

};


}}}//namespace simpla { namespace manifold{ namespace mertic

#endif //SIMPLA_CYLINDRICAL_H
