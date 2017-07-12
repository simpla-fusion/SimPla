//
// Created by salmon on 17-4-24.
//

#ifndef SIMPLA_SMESH_H
#define SIMPLA_SMESH_H

#include <simpla/algebra/all.h>
#include <simpla/data/all.h>
#include <simpla/engine/Domain.h>
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
    DOMAIN_POLICY_HEAD(SMesh);

    void InitialCondition(Real time_now);
    void BoundaryCondition(Real time_now, Real time_dt);

   public:
    //    point_type local_coordinates(index_type x, index_type y, index_type z, Real const *r) const override;

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
//
// template <typename THost>
// point_type SMesh<THost>::local_coordinates(index_type x, index_type y, index_type z, Real const *r) const {
//    return point_type{static_cast<Real>(x) + r[0], static_cast<Real>(y) + r[1], static_cast<Real>(z) + r[2]};
//}

/**
*\verbatim
*                ^s (dl)
*               /
*   (dz) t     /
*        ^    /
*        |   6---------------7
*        |  /|              /|
*        | / |             / |
*        |/  |            /  |
*        4---|-----------5   |
*        | m |           |   |
*        |   2-----------|----3
*        |  /            |  /
*        | /             | /
*        |/              |/
*        0---------------1---> r (dr)
*
*\endverbatim
*/
template <typename THost>
Real QuadrilateralArea(SMesh<THost> const *m, EntityId s, int d) {
    static constexpr int16_t r[3][4][3] = {{{0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {0, 1, 1}},
                                           {{0, 0, 0}, {1, 0, 0}, {0, 0, 1}, {1, 0, 1}},
                                           {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 1, 0}}};

    auto p0 = m->point(EntityId{s.x, s.y, s.z, s.w});
    auto p1 = m->point(EntityId{static_cast<int16_t>(s.x + r[d][1][0]), static_cast<int16_t>(s.y + r[d][1][1]),
                                static_cast<int16_t>(s.z + r[d][1][2]), s.w});
    auto p2 = m->point(EntityId{static_cast<int16_t>(s.x + r[d][2][0]), static_cast<int16_t>(s.y + r[d][2][1]),
                                static_cast<int16_t>(s.z + r[d][2][2]), s.w});
    auto p3 = m->point(EntityId{static_cast<int16_t>(s.x + r[d][3][0]), static_cast<int16_t>(s.y + r[d][3][1]),
                                static_cast<int16_t>(s.z + r[d][3][2]), s.w});

    return m->GetChart()->area(p0, p1, p2) + m->GetChart()->area(p1, p2, p3);
}

template <typename THost>
Real HexahedronVolume(SMesh<THost> const *m, EntityId s) {
    static constexpr int16_t r[8][3] = {
        {0, 0, 0},  //
        {0, 0, 1},  //
        {0, 1, 0},  //
        {0, 1, 1},  //
        {1, 0, 0},  //
        {1, 0, 1},  //
        {1, 1, 0},  //
        {1, 1, 1}   //

    };

    auto p0 = m->point(EntityId{s.x, s.y, s.z, s.w});
    auto p1 = m->point(EntityId{static_cast<int16_t>(s.x + r[1][0]), static_cast<int16_t>(s.y + r[1][1]),
                                static_cast<int16_t>(s.z + r[1][2]), s.w});
    auto p2 = m->point(EntityId{static_cast<int16_t>(s.x + r[2][0]), static_cast<int16_t>(s.y + r[2][1]),
                                static_cast<int16_t>(s.z + r[2][2]), s.w});
    auto p3 = m->point(EntityId{static_cast<int16_t>(s.x + r[3][0]), static_cast<int16_t>(s.y + r[3][1]),
                                static_cast<int16_t>(s.z + r[3][2]), s.w});
    auto p4 = m->point(EntityId{static_cast<int16_t>(s.x + r[4][0]), static_cast<int16_t>(s.y + r[4][1]),
                                static_cast<int16_t>(s.z + r[4][2]), s.w});
    auto p5 = m->point(EntityId{static_cast<int16_t>(s.x + r[5][0]), static_cast<int16_t>(s.y + r[5][1]),
                                static_cast<int16_t>(s.z + r[5][2]), s.w});
    auto p6 = m->point(EntityId{static_cast<int16_t>(s.x + r[6][0]), static_cast<int16_t>(s.y + r[6][1]),
                                static_cast<int16_t>(s.z + r[6][2]), s.w});
    auto p7 = m->point(EntityId{static_cast<int16_t>(s.x + r[7][0]), static_cast<int16_t>(s.y + r[7][1]),
                                static_cast<int16_t>(s.z + r[7][2]), s.w});

    return m->GetChart()->volume(p0, p1, p2, p4) + m->GetChart()->volume(p1, p2, p4, p6) +
           m->GetChart()->volume(p4, p6, p1, p5) + m->GetChart()->volume(p6, p1, p5, p7) +
           m->GetChart()->volume(p1, p5, p7, p2) + m->GetChart()->volume(p5, p7, p2, p3);
}

template <typename THost>
void SMesh<THost>::InitialCondition(Real time_now) {
    m_coordinates_ = [&](EntityId s) -> point_type { return global_coordinates(s, nullptr); };

    m_vertex_volume_.Clear();
    m_vertex_inv_volume_.Clear();
    m_vertex_dual_volume_.Clear();
    m_vertex_inv_dual_volume_.Clear();
    m_volume_volume_.Clear();
    m_volume_inv_volume_.Clear();
    m_volume_dual_volume_.Clear();
    m_volume_inv_dual_volume_.Clear();
    m_edge_volume_.Clear();
    m_edge_inv_volume_.Clear();
    m_edge_dual_volume_.Clear();
    m_edge_inv_dual_volume_.Clear();
    m_face_volume_.Clear();
    m_face_inv_volume_.Clear();
    m_face_dual_volume_.Clear();
    m_face_inv_dual_volume_.Clear();

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
    auto chart = GetChart();

    m_vertex_volume_ = 1.0;
    m_vertex_inv_volume_ = 1.0;
    m_vertex_dual_volume_ = [&](EntityId s) -> Real {
        return HexahedronVolume(this, EntityId{static_cast<int16_t>(s.x - 1), static_cast<int16_t>(s.y - 1),
                                               static_cast<int16_t>(s.z - 1), 0b111});
    };
    m_vertex_inv_dual_volume_ = 1.0 / m_vertex_dual_volume_;

    m_volume_volume_ = [&](EntityId s) -> Real {
        return HexahedronVolume(this, EntityId{s.x, s.y, s.z, 0b0});

    };
    m_volume_inv_volume_ = 1.0 / m_volume_volume_;
    m_volume_dual_volume_ = 1.0;
    m_volume_inv_dual_volume_ = 1.0;

    m_edge_volume_ = [&](EntityId s) -> Real {
        return chart->length(point(EntityId{s.x, s.y, s.z, 0b0}),
                             point(EntityId{static_cast<int16_t>(s.x + (s.w & 0b111) == 0b001 ? 1 : 0),
                                            static_cast<int16_t>(s.y + (s.w & 0b111) == 0b010 ? 1 : 0),
                                            static_cast<int16_t>(s.z + (s.w & 0b111) == 0b100 ? 1 : 0), 0b0}));
    };
    m_edge_inv_volume_ = 1.0 / m_edge_volume_;

    m_edge_dual_volume_ = [&](EntityId s) -> Real {
        int n = EntityIdCoder::m_id_to_sub_index_[s.w];
        return QuadrilateralArea(
            this, EntityId{static_cast<int16_t>(s.x - (n != 0 ? 1 : 0)), static_cast<int16_t>(s.y - (n != 1 ? 1 : 0)),
                           static_cast<int16_t>(s.z - (n != 2 ? 1 : 0)), 0b111},
            n);
    };
    m_edge_inv_dual_volume_ = 1.0 / m_edge_dual_volume_;

    m_face_volume_ = [&](EntityId s) -> Real {
        return QuadrilateralArea(this, EntityId{s.x, s.y, s.z, 0b0}, EntityIdCoder::m_id_to_sub_index_[s.w]);
    };

    m_face_inv_volume_ = 1.0 / m_face_volume_;
    m_face_dual_volume_ = [&](EntityId s) -> Real {
        return chart->length(point(EntityId{static_cast<int16_t>(s.x - (s.w & 0b111) == 0b110 ? 1 : 0),
                                            static_cast<int16_t>(s.y - (s.w & 0b111) == 0b101 ? 1 : 0),
                                            static_cast<int16_t>(s.z - (s.w & 0b111) == 0b011 ? 1 : 0), 0b111}),
                             point(EntityId{s.x, s.y, s.z, 0b111}));
    };
    m_face_inv_dual_volume_ = 1.0 / m_face_dual_volume_;
};

template <typename THost>
void SMesh<THost>::BoundaryCondition(Real time_now, Real time_dt) {
    m_host_->FillRange("PATCH_BOUNDARY_", m_vertex_volume_, 0);
    m_host_->FillRange("PATCH_BOUNDARY_", m_vertex_dual_volume_, 0);
    m_host_->FillRange("PATCH_BOUNDARY_", m_edge_volume_, 0);
    m_host_->FillRange("PATCH_BOUNDARY_", m_edge_dual_volume_, 0);
    m_host_->FillRange("PATCH_BOUNDARY_", m_face_volume_, 0);
    m_host_->FillRange("PATCH_BOUNDARY_", m_face_dual_volume_, 0);
    m_host_->FillRange("PATCH_BOUNDARY_", m_volume_volume_, 0);
    m_host_->FillRange("PATCH_BOUNDARY_", m_volume_dual_volume_, 0);
}

}  // namespace mesh {
}  // namespace simpla {
#endif  // SIMPLA_SMESH_H
