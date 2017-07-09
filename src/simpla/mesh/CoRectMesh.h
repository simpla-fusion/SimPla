//
// Created by salmon on 17-4-11.
//

#ifndef SIMPLA_CORECTMESH_H
#define SIMPLA_CORECTMESH_H

#include <simpla/engine/Domain.h>
#include <simpla/utilities/ObjectHead.h>
#include <string>
#include "StructuredMesh.h"
namespace simpla {
namespace mesh {
/**
 * Axis are perpendicular and spacing is constant
 */
struct CoRectMesh : public engine::Domain, public StructuredMesh {
    SP_OBJECT_HEAD(CoRectMesh, engine::Domain)

   public:
    template <typename... Args>
    explicit CoRectMesh(Args &&... args) : engine::Domain() {};
    ~CoRectMesh() override = default;

    SP_DEFAULT_CONSTRUCT(CoRectMesh)
    DECLARE_REGISTER_NAME(CoRectMesh);

    void DoInitialCondition(Real time_now) override;

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

}  // namespace  mesh
}  // namespace simpla

#endif  // SIMPLA_CORECTMESH_H
