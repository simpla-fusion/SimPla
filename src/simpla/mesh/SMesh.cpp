//
// Created by salmon on 17-4-25.
//

#include "SMesh.h"
#include <simpla/utilities/EntityIdCoder.h>
#include "StructuredMesh.h"
namespace simpla {
namespace mesh {
REGISTER_CREATOR(SMesh);

point_type SMesh::point(EntityId s) const { return local_coordinates(s, point_type{0, 0, 0}); }

point_type SMesh::local_coordinates(EntityId s, point_type const &pr) const {
    s.w = 0;
    point_type r{
        EntityIdCoder::m_id_to_coordinates_shift_[s.w & 0b111][0] + pr[0],
        EntityIdCoder::m_id_to_coordinates_shift_[s.w & 0b111][1] + pr[1],
        EntityIdCoder::m_id_to_coordinates_shift_[s.w & 0b111][2] + pr[2],
    };
    return point_type{std::fma(static_cast<Real>(s.x), m_dx_[0], r[0] * m_dx_[0]),
                      std::fma(static_cast<Real>(s.y), m_dx_[1], r[1] * m_dx_[1]),
                      std::fma(static_cast<Real>(s.z), m_dx_[2], r[2] * m_dx_[2])};
}

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

Real QuadrilateralArea(SMesh const *m, EntityId s, int d) {
    static constexpr int16_t r[3][4][3] = {{{0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {0, 1, 1}},
                                           {{0, 0, 0}, {1, 0, 0}, {0, 0, 1}, {1, 0, 1}},
                                           {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 1, 0}}};

    auto p0 = m->point(EntityId{.w = s.w, .x = s.x, .y = s.y, .z = s.z});
    auto p1 = m->point(EntityId{.w = s.w,
                                .x = static_cast<int16_t>(s.x + r[d][1][0]),
                                .y = static_cast<int16_t>(s.y + r[d][1][1]),
                                .z = static_cast<int16_t>(s.z + r[d][1][2])});
    auto p2 = m->point(EntityId{.w = s.w,
                                .x = static_cast<int16_t>(s.x + r[d][2][0]),
                                .y = static_cast<int16_t>(s.y + r[d][2][1]),
                                .z = static_cast<int16_t>(s.z + r[d][2][2])});
    auto p3 = m->point(EntityId{.w = s.w,
                                .x = static_cast<int16_t>(s.x + r[d][3][0]),
                                .y = static_cast<int16_t>(s.y + r[d][3][1]),
                                .z = static_cast<int16_t>(s.z + r[d][3][2])});

    return m->GetChart()->area(p0, p1, p2) + m->GetChart()->area(p1, p2, p3);
}

Real HexahedronVolume(SMesh const *m, EntityId s) {
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

    auto p0 = m->point(EntityId{.w = s.w, .x = s.x, .y = s.y, .z = s.z});
    auto p1 = m->point(EntityId{.w = s.w,
                                .x = static_cast<int16_t>(s.x + r[1][0]),
                                .y = static_cast<int16_t>(s.y + r[1][1]),
                                .z = static_cast<int16_t>(s.z + r[1][2])});
    auto p2 = m->point(EntityId{.w = s.w,
                                .x = static_cast<int16_t>(s.x + r[2][0]),
                                .y = static_cast<int16_t>(s.y + r[2][1]),
                                .z = static_cast<int16_t>(s.z + r[2][2])});
    auto p3 = m->point(EntityId{.w = s.w,
                                .x = static_cast<int16_t>(s.x + r[3][0]),
                                .y = static_cast<int16_t>(s.y + r[3][1]),
                                .z = static_cast<int16_t>(s.z + r[3][2])});
    auto p4 = m->point(EntityId{.w = s.w,
                                .x = static_cast<int16_t>(s.x + r[4][0]),
                                .y = static_cast<int16_t>(s.y + r[4][1]),
                                .z = static_cast<int16_t>(s.z + r[4][2])});
    auto p5 = m->point(EntityId{.w = s.w,
                                .x = static_cast<int16_t>(s.x + r[5][0]),
                                .y = static_cast<int16_t>(s.y + r[5][1]),
                                .z = static_cast<int16_t>(s.z + r[5][2])});
    auto p6 = m->point(EntityId{.w = s.w,
                                .x = static_cast<int16_t>(s.x + r[6][0]),
                                .y = static_cast<int16_t>(s.y + r[6][1]),
                                .z = static_cast<int16_t>(s.z + r[6][2])});
    auto p7 = m->point(EntityId{.w = s.w,
                                .x = static_cast<int16_t>(s.x + r[7][0]),
                                .y = static_cast<int16_t>(s.y + r[7][1]),
                                .z = static_cast<int16_t>(s.z + r[7][2])});

    return m->GetChart()->volume(p0, p1, p2, p4) + m->GetChart()->volume(p1, p2, p4, p6) +
            m->GetChart()->volume(p4, p6, p1, p5) + m->GetChart()->volume(p6, p1, p5, p7) +
            m->GetChart()->volume(p1, p5, p7, p2) + m->GetChart()->volume(p5, p7, p2, p3);
}

void SMesh::InitializeData(Real time_now) {
    StructuredMesh::InitializeData(time_now);

    m_coordinates_ = [&](EntityId s) -> point_type { return global_coordinates(s); };

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
        return HexahedronVolume(this, EntityId{.w = 0b111,
                                               .x = static_cast<int16_t>(s.x - 1),
                                               .y = static_cast<int16_t>(s.y - 1),
                                               .z = static_cast<int16_t>(s.z - 1)});

    };
    m_vertex_inv_dual_volume_ = 1.0 / m_vertex_dual_volume_;

    m_volume_volume_ = [&](EntityId s) -> Real {
        return HexahedronVolume(this, EntityId{.w = 0b0, .x = s.x, .y = s.y, .z = s.z});

    };
    m_volume_inv_volume_ = 1.0 / m_volume_volume_;
    m_volume_dual_volume_ = 1.0;
    m_volume_inv_dual_volume_ = 1.0;

    m_edge_volume_ = [&](EntityId s) -> Real {
        return chart->length(point(EntityId{.w = 0b0, .x = s.x, .y = s.y, .z = s.z}),
                             point(EntityId{.w = 0b0,
                                            .x = static_cast<int16_t>(s.x + (s.w & 0b111) == 0b001 ? 1 : 0),
                                            .y = static_cast<int16_t>(s.y + (s.w & 0b111) == 0b010 ? 1 : 0),
                                            .z = static_cast<int16_t>(s.z + (s.w & 0b111) == 0b100 ? 1 : 0)}));
    };
    m_edge_inv_volume_ = 1.0 / m_edge_volume_;

    m_edge_dual_volume_ = [&](EntityId s) -> Real {
        int n = EntityIdCoder::m_id_to_sub_index_[s.w];
        return QuadrilateralArea(this, EntityId{.w = 0b111,
                                                .x = static_cast<int16_t>(s.x - (n != 0 ? 1 : 0)),
                                                .y = static_cast<int16_t>(s.y - (n != 1 ? 1 : 0)),
                                                .z = static_cast<int16_t>(s.z - (n != 2 ? 1 : 0))},
                                 n);
    };
    m_edge_inv_dual_volume_ = 1.0 / m_edge_dual_volume_;

    m_face_volume_ = [&](EntityId s) -> Real {
        return QuadrilateralArea(this, EntityId{.w = 0b0, .x = s.x, .y = s.y, .z = s.z},
                                 EntityIdCoder::m_id_to_sub_index_[s.w]);
    };

    m_face_inv_volume_ = 1.0 / m_face_volume_;
    m_face_dual_volume_ = [&](EntityId s) -> Real {
        return chart->length(point(EntityId{.w = 0b111,
                                            .x = static_cast<int16_t>(s.x - (s.w & 0b111) == 0b110 ? 1 : 0),
                                            .y = static_cast<int16_t>(s.y - (s.w & 0b111) == 0b101 ? 1 : 0),
                                            .z = static_cast<int16_t>(s.z - (s.w & 0b111) == 0b011 ? 1 : 0)}),
                             point(EntityId{.w = 0b111, .x = s.x, .y = s.y, .z = s.z}));
    };
    m_face_inv_dual_volume_ = 1.0 / m_face_dual_volume_;
};

void SMesh::SetBoundaryCondition(Real time_now, Real time_dt) {
    StructuredMesh::SetBoundaryCondition(time_now, time_dt);

    m_vertex_volume_[GetRange("VERTEX_PATCH_BOUNDARY")] = 0;
    m_vertex_dual_volume_[GetRange("VERTEX_PATCH_BOUNDARY")] = 0;

    m_edge_volume_[GetRange("EDGE_PATCH_BOUNDARY")] = 0;
    m_edge_dual_volume_[GetRange("EDGE_PATCH_BOUNDARY")] = 0;

    m_face_volume_[GetRange("FACE_PATCH_BOUNDARY")] = 0;
    m_face_dual_volume_[GetRange("FACE_PATCH_BOUNDARY")] = 0;

    m_volume_volume_[GetRange("VOLUME_PATCH_BOUNDARY")] = 0;
    m_volume_dual_volume_[GetRange("VOLUME_PATCH_BOUNDARY")] = 0;
}
}
}