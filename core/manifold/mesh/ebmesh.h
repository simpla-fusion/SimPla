/**
 * @file ebmesh.h.h
 * @author salmon
 * @date 2015-12-08.
 */

#ifndef SIMPLA_EBMESH_H_H
#define SIMPLA_EBMESH_H_H

namespace simpla
{
namespace tags { struct embedded; }

template<typename TMesh>
struct Manifold<TMesh, tags::embedded> : public TMesh
{


};

}


#endif //SIMPLA_EBMESH_H_H
