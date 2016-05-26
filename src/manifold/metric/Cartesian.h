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

    mesh_type const &m_;

    typedef typename mesh::MeshEntityId id_type;
public:
    typedef this_type metric_policy;

    Cartesian(mesh_type const &m) : m_(m) { }

    ~Cartesian() { }

public:

private:
    Real m_volume_[9];
    Real m_inv_volume_[9];
    Real m_dual_volume_[9];
    Real m_inv_dual_volume_[9];
public:


    virtual Real volume(id_type s) const { return m_volume_[mesh_type::node_id(s)]; }

    virtual Real dual_volume(id_type s) const { return m_dual_volume_[mesh_type::node_id(s)]; }

    virtual Real inv_volume(id_type s) const { return m_inv_volume_[mesh_type::node_id(s)]; }

    virtual Real inv_dual_volume(id_type s) const { return m_inv_dual_volume_[mesh_type::node_id(s)]; }
};

}}}//namespace simpla { namespace manifold{ namespace mertic

#endif //SIMPLA_CARTESIAN_H
