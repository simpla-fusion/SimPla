//
// Created by salmon on 17-6-1.
//

#ifndef SIMPLA_RECTMESH_H
#define SIMPLA_RECTMESH_H
#include <simpla/algebra/all.h>
#include <simpla/data/all.h>
#include <simpla/engine/Domain.h>
#include "StructuredMesh.h"
namespace simpla {
namespace mesh {

using namespace simpla::data;
/**
 * Axis are perpendicular
 */
template <typename THost>
struct RectMesh : public StructuredMesh {
    DOMAIN_POLICY_HEAD(RectMesh);

   public:
    void InitialCondition(Real time_now);
    void BoundaryCondition(Real time_now, Real time_dt);

    Field<host_type, Real, VERTEX, 3> m_coordinates_{m_host_, "name"_ = "m_coordinates_", "COORDINATES"_};
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
};
// template <typename THost>
// point_type RectMesh<THost>::local_coordinates(EntityId s, Real const *pr) const {
//    point_type r{
//        (EntityIdCoder::m_id_to_coordinates_shift_[s.w & 0b111][0] + ((pr == nullptr) ? 0 : (pr[0] * m_dx_[0]))),
//        (EntityIdCoder::m_id_to_coordinates_shift_[s.w & 0b111][1] + ((pr == nullptr) ? 0 : (pr[1] * m_dx_[1]))),
//        (EntityIdCoder::m_id_to_coordinates_shift_[s.w & 0b111][2] + ((pr == nullptr) ? 0 : (pr[2] * m_dx_[2]))),
//    };
//    return point_type{std::fma(static_cast<Real>(s.x), m_dx_[0], r[0]),
//                      std::fma(static_cast<Real>(s.y), m_dx_[1], r[1]),
//                      std::fma(static_cast<Real>(s.z), m_dx_[2], r[2])};
//}

template <typename THost>
void RectMesh<THost>::InitialCondition(Real time_now) {
    m_coordinates_ = [&](point_type const &x) -> point_type { return map(x); };
    m_vertices_ = [&](point_type const &x) -> point_type { return (x); };
    m_vertex_volume_.Initialize();
    m_vertex_inv_volume_.Initialize();
    m_vertex_dual_volume_.Initialize();
    m_vertex_inv_dual_volume_.Initialize();

    m_volume_volume_.Initialize();
    m_volume_inv_volume_.Initialize();
    m_volume_dual_volume_.Initialize();
    m_volume_inv_dual_volume_.Initialize();

    m_edge_volume_.Initialize();
    m_edge_inv_volume_.Initialize();
    m_edge_dual_volume_.Initialize();
    m_edge_inv_dual_volume_.Initialize();

    m_face_volume_.Initialize();
    m_face_inv_volume_.Initialize();
    m_face_dual_volume_.Initialize();
    m_face_inv_dual_volume_.Initialize();

    /**
        *\verbatim
        *                ^y (dl)
        *               /
        *   (dz) z     /
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
        *       000-------------001---> x (dr)
        *
        *\endverbatim
        */

    auto chart = m_host_->GetChart();
    m_vertex_volume_ = 1.0;
    m_vertex_inv_volume_ = 1.0;
    m_vertex_dual_volume_ = [&](index_type x, index_type y, index_type z, int tag) -> Real {
        return chart->volume(global_coordinates(x - 1, y - 1, z - 1, 0b111), global_coordinates(x, y, z, 0b111));
    };
    m_vertex_inv_dual_volume_ = 1.0 / m_vertex_dual_volume_;

    m_volume_volume_ = [&](index_type x, index_type y, index_type z, int tag) -> Real {
        return chart->volume(global_coordinates(x, y, z, 0b0), global_coordinates(x + 1, y + 1, z + 1, 0b0));
    };
    m_volume_inv_volume_ = 1.0 / m_volume_volume_;
    m_volume_dual_volume_ = 1.0;
    m_volume_inv_dual_volume_ = 1.0;

    m_edge_volume_ = [&](EntityId s) -> Real {
        return chart->length(point(EntityId{s.x, s.y, s.z, 0b0}),
                             point(EntityId{static_cast<int16_t>(s.x + ((s.w & 0b111) == 0b001 ? 1 : 0)),
                                            static_cast<int16_t>(s.y + ((s.w & 0b111) == 0b010 ? 1 : 0)),
                                            static_cast<int16_t>(s.z + ((s.w & 0b111) == 0b100 ? 1 : 0)), 0b0}),
                             EntityIdCoder::m_id_to_sub_index_[s.w & 0b111]);
    };
    m_edge_inv_volume_ = 1.0 / m_edge_volume_;

    m_edge_dual_volume_ = [&](EntityId s) -> Real {
        return chart->area(point(EntityId{static_cast<int16_t>(s.x - ((s.w & 0b111) != 0b001 ? 1 : 0)),
                                          static_cast<int16_t>(s.y - ((s.w & 0b111) != 0b010 ? 1 : 0)),
                                          static_cast<int16_t>(s.z - ((s.w & 0b111) != 0b100 ? 1 : 0)), 0b111}),
                           point(EntityId{s.x, s.y, s.z, 0b111}), EntityIdCoder::m_id_to_sub_index_[s.w & 0b111]);
    };
    m_edge_inv_dual_volume_ = 1.0 / m_edge_dual_volume_;

    m_face_volume_ = [&](EntityId s) -> Real {
        return chart->area(point(EntityId{s.x, s.y, s.z, 0b0}),
                           point(EntityId{static_cast<int16_t>(s.x + ((s.w & 0b111) != 0b110 ? 1 : 0)),
                                          static_cast<int16_t>(s.y + ((s.w & 0b111) != 0b101 ? 1 : 0)),
                                          static_cast<int16_t>(s.z + ((s.w & 0b111) != 0b011 ? 1 : 0)), 0b0}),
                           EntityIdCoder::m_id_to_sub_index_[s.w & 0b111]);

    };

    m_face_inv_volume_ = 1.0 / m_face_volume_;

    m_face_dual_volume_ = [&](EntityId s) -> Real {
        return chart->length(point(EntityId{static_cast<int16_t>(s.x - ((s.w & 0b111) == 0b110 ? 1 : 0)),
                                            static_cast<int16_t>(s.y - ((s.w & 0b111) == 0b101 ? 1 : 0)),
                                            static_cast<int16_t>(s.z - ((s.w & 0b111) == 0b011 ? 1 : 0)), 0b111}),
                             point(EntityId{s.x, s.y, s.z, 0b111}), EntityIdCoder::m_id_to_sub_index_[s.w & 0b111]);
    };
    m_face_inv_dual_volume_ = 1.0 / m_face_dual_volume_;
};

template <typename THost>
void RectMesh<THost>::BoundaryCondition(Real time_now, Real time_dt) {
    m_host_->FillBoundary(m_vertex_volume_, 0);
    m_host_->FillBoundary(m_vertex_dual_volume_, 0);
    m_host_->FillBoundary(m_vertex_inv_volume_, 0);
    m_host_->FillBoundary(m_vertex_inv_dual_volume_, 0);

    m_host_->FillBoundary(m_edge_volume_, 0);
    m_host_->FillBoundary(m_edge_dual_volume_, 0);
    m_host_->FillBoundary(m_edge_inv_volume_, 0);
    m_host_->FillBoundary(m_edge_inv_dual_volume_, 0);

    m_host_->FillBoundary(m_face_volume_, 0);
    m_host_->FillBoundary(m_face_dual_volume_, 0);
    m_host_->FillBoundary(m_face_inv_volume_, 0);
    m_host_->FillBoundary(m_face_inv_dual_volume_, 0);

    m_host_->FillBoundary(m_volume_volume_, 0);
    m_host_->FillBoundary(m_volume_dual_volume_, 0);
    m_host_->FillBoundary(m_volume_inv_volume_, 0);
    m_host_->FillBoundary(m_volume_inv_dual_volume_, 0);
}

}  // namespace mesh {
}  // namespace simpla {
#endif  // SIMPLA_RECTMESH_H
