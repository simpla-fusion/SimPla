//
// Created by salmon on 17-4-11.
//

#ifndef SIMPLA_CORECTMESH_H
#define SIMPLA_CORECTMESH_H

#include <string>
#include "StructuredMesh.h"
#include "simpla/data/Data.h"
#include "simpla/engine/Domain.h"
namespace simpla {
namespace mesh {
/**
 * Axis are perpendicular and spacing is constant
 */
template <typename THost>
struct CoRectMesh : public StructuredMesh {
    SP_DOMAIN_POLICY_HEAD(CoRectMesh);

   public:
    void InitialCondition(Real time_now);

    Real m_node_volume_ = 1.0;
    Real m_node_inv_volume_ = 1.0;
    Real m_node_dual_volume_ = 1.0;
    Real m_node_inv_dual_volume_ = 1.0;

    Real m_cell_volume_ = 1.0;
    Real m_cell_inv_volume_ = 1.0;
    Real m_cell_dual_volume_ = 1.0;
    Real m_cell_inv_dual_volume_ = 1.0;

    Real m_edge_volume_[3] = {1, 1, 1};
    Real m_edge_inv_volume_[3] = {1, 1, 1};
    Real m_edge_dual_volume_[3] = {1, 1, 1};
    Real m_edge_inv_dual_volume_[3] = {1, 1, 1};

    Real m_face_volume_[3] = {1, 1, 1};
    Real m_face_inv_volume_[3] = {1, 1, 1};
    Real m_face_dual_volume_[3] = {1, 1, 1};
    Real m_face_inv_dual_volume_[3] = {1, 1, 1};

};  // struct  MeshBase
template <typename THost>
CoRectMesh<THost>::CoRectMesh(THost* h) : m_host_(h) {
    h->PreInitialCondition.Connect([=](engine::DomainBase* self, Real time_now) {
        if (auto* p = dynamic_cast<CoRectMesh<THost>*>(self)) { p->InitialCondition(time_now); }
    });
    h->OnSerialize.Connect([=](engine::DomainBase const* self, std::shared_ptr<simpla::data::DataNode>& tdb) {
        if (auto const* p = dynamic_cast<CoRectMesh<THost> const*>(self)) { tdb->Set(p->Serialize()); }
    });
}
template <typename THost>
CoRectMesh<THost>::~CoRectMesh() {}

template <typename THost>
std::shared_ptr<data::DataNode> CoRectMesh<THost>::Serialize() const {
    auto res = data::DataNode::New(data::DataNode::DN_TABLE);
    res->SetValue("Topology", "3DCoRectMesh");
    return res;
}
template <typename THost>
void CoRectMesh<THost>::Deserialize(std::shared_ptr<data::DataNode> const& cfg) {}
template <typename THost>
void CoRectMesh<THost>::InitialCondition(Real time_now) {
    //    Update();
    /**
     *\verbatim
     *                ^y
     *               /
     *        z     /
     *        ^    /
     *        |  110-------------111
     *        |  /|              /|
     *        | / |             / |
     *        |/  |            /  |
     *       100--|----------101  |
     *        | m |           |   |
     *        |  010----------|--011
     *        |  /            |  /
     *        | /             | /
     *        |/              |/
     *       000-------------001---> x
     *
     *\endverbatim
     */
    //    m_x0_ = GetBaseChart()->GetOrigin();
    //    m_coarsest_cell_width_ = GetBaseChart()->GetDx();
    //    size_tuple m_dims_ = GetBlock().GetDimensions();

    //    m_cell_[0 /*000*/] = 1;
    //    m_cell_[1 /*001*/] = (m_dims_[0] == 1) ? 1 : m_coarsest_cell_width_[0];
    //    m_cell_[2 /*010*/] = (m_dims_[1] == 1) ? 1 : m_coarsest_cell_width_[1];
    //    m_cell_[4 /*100*/] = (m_dims_[2] == 1) ? 1 : m_coarsest_cell_width_[2];
    //    m_cell_[3 /*011*/] = m_cell_[1] * m_cell_[2];
    //    m_cell_[5 /*101*/] = m_cell_[4] * m_cell_[1];
    //    m_cell_[6 /*110*/] = m_cell_[4] * m_cell_[2];
    //    m_cell_[7 /*111*/] = m_cell_[1] * m_cell_[2] * m_cell_[4];
    //
    //    m_dual_volume_[0 /*000*/] = m_cell_[7];
    //    m_dual_volume_[1 /*001*/] = m_cell_[6];
    //    m_dual_volume_[2 /*010*/] = m_cell_[5];
    //    m_dual_volume_[4 /*100*/] = m_cell_[3];
    //    m_dual_volume_[3 /*011*/] = m_cell_[4];
    //    m_dual_volume_[5 /*101*/] = m_cell_[2];
    //    m_dual_volume_[6 /*110*/] = m_cell_[1];
    //    m_dual_volume_[7 /*111*/] = m_cell_[0];
    //
    //    m_inv_volume_[0 /*000*/] = 1;
    //    m_inv_volume_[1 /*001*/] = (m_dims_[0] == 1) ? 1 : m_inv_dx_[0];
    //    m_inv_volume_[2 /*010*/] = (m_dims_[1] == 1) ? 1 : m_inv_dx_[1];
    //    m_inv_volume_[4 /*100*/] = (m_dims_[2] == 1) ? 1 : m_inv_dx_[2];
    //    m_inv_volume_[3 /*011*/] = m_inv_volume_[2] * m_inv_volume_[1];
    //    m_inv_volume_[5 /*101*/] = m_inv_volume_[4] * m_inv_volume_[1];
    //    m_inv_volume_[6 /*110*/] = m_inv_volume_[4] * m_inv_volume_[2];
    //    m_inv_volume_[7 /*111*/] = m_inv_volume_[1] * m_inv_volume_[2] * m_inv_volume_[4];
    //
    //    m_inv_volume_[1 /*001*/] = (m_dims_[0] == 1) ? 0 : m_inv_volume_[1];
    //    m_inv_volume_[2 /*010*/] = (m_dims_[1] == 1) ? 0 : m_inv_volume_[2];
    //    m_inv_volume_[4 /*100*/] = (m_dims_[2] == 1) ? 0 : m_inv_volume_[4];
    //
    //    m_inv_dual_volume_[0 /*000*/] = m_inv_volume_[7];
    //    m_inv_dual_volume_[1 /*001*/] = m_inv_volume_[6];
    //    m_inv_dual_volume_[2 /*010*/] = m_inv_volume_[5];
    //    m_inv_dual_volume_[4 /*100*/] = m_inv_volume_[3];
    //    m_inv_dual_volume_[3 /*011*/] = m_inv_volume_[4];
    //    m_inv_dual_volume_[5 /*101*/] = m_inv_volume_[2];
    //    m_inv_dual_volume_[6 /*110*/] = m_inv_volume_[1];
    //    m_inv_dual_volume_[7 /*111*/] = m_inv_volume_[0];
}

}  // namespace  mesh
}  // namespace simpla

#endif  // SIMPLA_CORECTMESH_H
