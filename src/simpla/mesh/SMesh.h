//
// Created by salmon on 17-4-24.
//

#ifndef SIMPLA_SMESH_H
#define SIMPLA_SMESH_H

#include <simpla/algebra/all.h>
#include <simpla/data/all.h>
#include <simpla/engine/DomainBase.h>
#include "Mesh.h"
#include "StructuredMesh.h"
namespace simpla {
namespace mesh {
using namespace simpla::data;

/**
 *  Curvilinear
 *  metric is not uniform
 */
template <typename THost>
struct SMesh : public StructuredMesh {
   public:
    SP_OBJECT_HEAD(SMesh, StructuredMesh)

    THost *m_host_;
    bool m_is_register_;
    typedef THost host_type;
    SMesh(host_type *host) : m_host_(host){};
    ~SMesh() override = default;

    SP_DEFAULT_CONSTRUCT(SMesh)

    void InitialCondition(Real time_now) override;
    void BoundaryCondition(Real time_now, Real time_dt) override;
    void Advance(Real time_now, Real time_dt) override;

    point_type local_coordinates(entity_id_type s, Real const *r) const override;

    Field<host_type, Real, VERTEX, 3> m_coordinates_{m_host_, "name"_ = "m_coordinates_" /*, "COORDINATES"_*/};
    Field<host_type, Real, VERTEX, 3> m_vertices_{m_host_, "name"_ = "m_vertices_"};

    Field<host_type, Real, VERTEX> m_vertex_volume_{m_host_, "name"_ = "m_vertex_volume_"};
    Field<host_type, Real, VERTEX> m_vertex_inv_volume_{m_host_, "name"_ = "m_vertex_inv_volume_"};
    Field<host_type, Real, VERTEX> m_vertex_dual_volume_{m_host_, "name"_ = "m_vertex_dual_volume_"};
    Field<host_type, Real, VERTEX> m_vertex_inv_dual_volume_{m_host_, "name"_ = "m_vertex_inv_dual_volume_"};
    Field<host_type, Real, VOLUME> m_volume_volume_{m_host_, "name"_ = "m_volume_volume_"};
    Field<host_type, Real, VOLUME> m_volume_inv_volume_{m_host_, "name"_ = "m_volume_inv_volume_"};
    Field<host_type, Real, VOLUME> m_volume_dual_volume_{m_host_, "name"_ = "m_volume_dual_volume_"};
    Field<host_type, Real, VOLUME> m_volume_inv_dual_volume_{m_host_, "name"_ = "m_volume_inv_dual_volume_"};
    Field<host_type, Real, EDGE> m_edge_volume_{m_host_, "name"_ = "m_edge_volume_"};
    Field<host_type, Real, EDGE> m_edge_inv_volume_{m_host_, "name"_ = "m_edge_inv_volume_"};
    Field<host_type, Real, EDGE> m_edge_dual_volume_{m_host_, "name"_ = "m_edge_dual_volume_"};
    Field<host_type, Real, EDGE> m_edge_inv_dual_volume_{m_host_, "name"_ = "m_edge_inv_dual_volume_"};
    Field<host_type, Real, FACE> m_face_volume_{m_host_, "name"_ = "m_face_volume_"};
    Field<host_type, Real, FACE> m_face_inv_volume_{m_host_, "name"_ = "m_face_inv_volume_"};
    Field<host_type, Real, FACE> m_face_dual_volume_{m_host_, "name"_ = "m_face_dual_volume_"};
    Field<host_type, Real, FACE> m_face_inv_dual_volume_{m_host_, "name"_ = "m_face_inv_dual_volume_"};
    Field<host_type, Real, VERTEX> m_vertex_hodge_{m_host_, "name"_ = "m_vertex_hodge_"};
    Field<host_type, Real, EDGE> m_edge_hodge_{m_host_, "name"_ = "m_edge_hodge_"};
    Field<host_type, Real, FACE> m_face_hodge_{m_host_, "name"_ = "m_face_hodge_"};
    Field<host_type, Real, VOLUME> m_volume_hodge_{m_host_, "name"_ = "m_volume_hodge_"};
};

}  // namespace mesh {
}  // namespace simpla {
#endif  // SIMPLA_SMESH_H
