//
// Created by salmon on 17-4-24.
//

#ifndef SIMPLA_SMESH_H
#define SIMPLA_SMESH_H

#include <simpla/algebra/all.h>
#include <simpla/data/all.h>
#include "StructuredMesh.h"
namespace simpla {
namespace mesh {
using namespace simpla::engine;

struct SMesh : public StructuredMesh {
   public:
    SP_OBJECT_HEAD(SMesh, StructuredMesh)

    explicit SMesh(Domain *d) : StructuredMesh(d){};
    ~SMesh() override = default;

    SP_DEFAULT_CONSTRUCT(SMesh)
    DECLARE_REGISTER_NAME("SMesh");
    void InitialCondition(Real time_now) override;
    void BoundaryCondition(Real time_now, Real time_dt) override;

    Field<this_type, Real, VERTEX, 3> m_coordinates_{this, "COORDINATES"_, "name"_ = "Coordinates"};
    Field<this_type, Real, VERTEX, 3> m_vertices_{this, "name"_ = "vertices"};
    //    Field<this_type, Real, VOLUME, 9> m_volume_{this, "name"_ = "volume"};
    //    Field<this_type, Real, VOLUME, 9> m_dual_volume_{this, "name"_ = "dual_volume"};
    //    Field<this_type, Real, VOLUME, 9> m_inv_volume_{this, "name"_ = "inv_volume"};
    //    Field<this_type, Real, VOLUME, 9> m_inv_dual_volume_{this, "name"_ = "inv_dual_volume"};

    Field<this_type, Real, VERTEX> m_vertex_volume_{this, "name"_ = "vertex_volume"};
    Field<this_type, Real, VERTEX> m_vertex_inv_volume_{this, "name"_ = "vertex_inv_volume"};
    Field<this_type, Real, VERTEX> m_vertex_dual_volume_{this, "name"_ = "vertex_dual_volume"};
    Field<this_type, Real, VERTEX> m_vertex_inv_dual_volume_{this, "name"_ = "vertex_inv_dual_volume"};

    Field<this_type, Real, VOLUME> m_volume_volume_{this, "name"_ = "volume_volume"};
    Field<this_type, Real, VOLUME> m_volume_inv_volume_{this, "name"_ = "volume_inv_volume"};
    Field<this_type, Real, VOLUME> m_volume_dual_volume_{this, "name"_ = "volume_dual_volume"};
    Field<this_type, Real, VOLUME> m_volume_inv_dual_volume_{this, "name"_ = "volume_inv_dual_volume"};

    Field<this_type, Real, EDGE> m_edge_volume_{this, "name"_ = "edge_volume"};
    Field<this_type, Real, EDGE> m_edge_inv_volume_{this, "name"_ = "edge_inv_volume"};
    Field<this_type, Real, EDGE> m_edge_dual_volume_{this, "name"_ = "edge_dual_volume"};
    Field<this_type, Real, EDGE> m_edge_inv_dual_volume_{this, "name"_ = "edge_inv_dual_volume"};

    Field<this_type, Real, FACE> m_face_volume_{this, "name"_ = "face_volume"};
    Field<this_type, Real, FACE> m_face_inv_volume_{this, "name"_ = "face_inv_volume"};
    Field<this_type, Real, FACE> m_face_dual_volume_{this, "name"_ = "face_dual_volume"};
    Field<this_type, Real, FACE> m_face_inv_dual_volume_{this, "name"_ = "face_inv_dual_volume"};

    point_type point(EntityId s) const override { return StructuredMesh::point(s); }

    //    Real volume(EntityId s) const override { return m_volume_[s]; }
    //    Real dual_volume(EntityId s) const override { return m_volume_[s]; }
    //    Real inv_volume(EntityId s) const override { return m_volume_[s]; }
    //    Real inv_dual_volume(EntityId s) const override { return m_volume_[s]; }
};

}  // namespace mesh {
}  // namespace simpla {
#endif  // SIMPLA_SMESH_H
