//
// Created by salmon on 17-4-11.
//

#ifndef SIMPLA_CORECTMESH_H
#define SIMPLA_CORECTMESH_H

#include <simpla/data/all.h>
#include <simpla/engine/Domain.h>
#include <string>
#include "StructuredMesh.h"
namespace simpla {
namespace mesh {
/**
 * Axis are perpendicular and spacing is constant
 */
template <typename THost>
struct CoRectMesh : public StructuredMesh {
    SP_OBJECT_HEAD(CoRectMesh, StructuredMesh)
    DOMAIN_POLICY_HEAD(CoRectMesh);

   public:
    Real m_vertex_volume_[1] = {1.0};
    Real m_vertex_inv_volume_[1] = {1.0};
    Real m_vertex_dual_volume_[1] = {1.0};
    Real m_vertex_inv_dual_volume_[1] = {1.0};

    Real m_volume_volume_[1] = {1.0};
    Real m_volume_inv_volume_[1] = {1.0};
    Real m_volume_dual_volume_[1] = {1.0};
    Real m_volume_inv_dual_volume_[1] = {1.0};

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
std::shared_ptr<data::DataTable> CoRectMesh<THost>::Serialize() const {
    return std::make_shared<data::DataTable>();
}

template <typename THost>
void CoRectMesh<THost>::Deserialize(std::shared_ptr<data::DataTable> const &cfg){};

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
    //    m_x0_ = GetChart()->GetOrigin();
    //    m_coarsest_cell_width_ = GetChart()->GetDx();
    size_tuple m_dims_ = GetBlock().GetDimensions();

    //    m_volume_[0 /*000*/] = 1;
    //    m_volume_[1 /*001*/] = (m_dims_[0] == 1) ? 1 : m_coarsest_cell_width_[0];
    //    m_volume_[2 /*010*/] = (m_dims_[1] == 1) ? 1 : m_coarsest_cell_width_[1];
    //    m_volume_[4 /*100*/] = (m_dims_[2] == 1) ? 1 : m_coarsest_cell_width_[2];
    //    m_volume_[3 /*011*/] = m_volume_[1] * m_volume_[2];
    //    m_volume_[5 /*101*/] = m_volume_[4] * m_volume_[1];
    //    m_volume_[6 /*110*/] = m_volume_[4] * m_volume_[2];
    //    m_volume_[7 /*111*/] = m_volume_[1] * m_volume_[2] * m_volume_[4];
    //
    //    m_dual_volume_[0 /*000*/] = m_volume_[7];
    //    m_dual_volume_[1 /*001*/] = m_volume_[6];
    //    m_dual_volume_[2 /*010*/] = m_volume_[5];
    //    m_dual_volume_[4 /*100*/] = m_volume_[3];
    //    m_dual_volume_[3 /*011*/] = m_volume_[4];
    //    m_dual_volume_[5 /*101*/] = m_volume_[2];
    //    m_dual_volume_[6 /*110*/] = m_volume_[1];
    //    m_dual_volume_[7 /*111*/] = m_volume_[0];
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

template <typename THost>
void CoRectMesh<THost>::BoundaryCondition(Real time_now, Real time_dt) {}

template <typename THost>
void CoRectMesh<THost>::Advance(Real time_now, Real time_dt) {}

}  // namespace  mesh
}  // namespace simpla

#endif  // SIMPLA_CORECTMESH_H
