//
// Created by salmon on 17-4-25.
//
#include "StructuredMesh.h"
#include <simpla/algebra/all.h>
#include <simpla/geometry/GeoAlgorithm.h>
#include <simpla/utilities/EntityId.h>
namespace simpla {
namespace mesh {
using namespace algebra;

void StructuredMesh::Update() {
    engine::MeshBase::Update();
    m_dx_ = engine::MeshBase::GetCellWidth();
    m_x0_ = engine::MeshBase::GetOrigin();
}

/**
*\verbatim
*                ^s (dl)
*               /
*   (dz) t     /
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
*       000-------------001---> r (dr)
*
*\endverbatim
*/

point_type StructuredMesh::local_coordinates(EntityId s, Real const *pr) const {
    ASSERT(pr != nullptr);
    point_type r;

    r[0] = pr[0] + EntityIdCoder::m_id_to_coordinates_shift_[s.w & 0b111][0];
    r[1] = pr[1] + EntityIdCoder::m_id_to_coordinates_shift_[s.w & 0b111][1];
    r[2] = pr[2] + EntityIdCoder::m_id_to_coordinates_shift_[s.w & 0b111][2];

    return point_type{std::fma(static_cast<Real>(s.x), m_dx_[0], r[0] * m_dx_[0] + m_x0_[0]),
                      std::fma(static_cast<Real>(s.y), m_dx_[1], r[1] * m_dx_[1] + m_x0_[1]),
                      std::fma(static_cast<Real>(s.z), m_dx_[2], r[2] * m_dx_[2] + m_x0_[2])};
}

point_type StructuredMesh::point(EntityId s) const {
    auto const *r = EntityIdCoder::m_id_to_coordinates_shift_[s.w & 0b111];
    return point_type{std::fma(static_cast<Real>(s.x), m_dx_[0], r[0] * m_dx_[0] + m_x0_[0]),
                      std::fma(static_cast<Real>(s.y), m_dx_[1], r[1] * m_dx_[1] + m_x0_[1]),
                      std::fma(static_cast<Real>(s.z), m_dx_[2], r[2] * m_dx_[2] + m_x0_[2])};
};

index_box_type StructuredMesh::GetIndexBox(int tag) const {
    index_box_type res = GetBlock().GetIndexBox();
    switch (tag) {
        case 0:
            std::get<1>(res) += 1;
            break;
        case 1:
            std::get<1>(res)[1] += 1;
            std::get<1>(res)[2] += 1;
            break;
        case 2:
            std::get<1>(res)[0] += 1;
            std::get<1>(res)[2] += 1;
            break;
        case 4:
            std::get<1>(res)[0] += 1;
            std::get<1>(res)[1] += 1;
            break;
        case 3:
            std::get<1>(res)[2] += 1;
            break;
        case 5:
            std::get<1>(res)[1] += 1;
            break;
        case 6:
            std::get<1>(res)[0] += 1;
            break;
        case 7:
        default:
            break;
    }
    return res;
}
void StructuredMesh::RegisterRanges(std::shared_ptr<geometry::GeoObject> const &g, std::string const &prefix) {
    Real ratio = g == nullptr ? 1.0 : g->CheckOverlap(GetBox());

    auto ranges = GetRanges();

    if (ratio < EPSILON) {  // no overlap

        (*ranges)[prefix + "." + std::string(EntityIFORMName[VERTEX]) + "_BODY"].append(
            std::make_shared<EmptyRangeBase<EntityId>>());

        (*ranges)[prefix + "." + std::string(EntityIFORMName[EDGE]) + "_BODY"].append(
            std::make_shared<EmptyRangeBase<EntityId>>());

        (*ranges)[prefix + "." + std::string(EntityIFORMName[FACE]) + "_BODY"].append(
            std::make_shared<EmptyRangeBase<EntityId>>());

        (*ranges)[prefix + "." + std::string(EntityIFORMName[VOLUME]) + "_BODY"].append(
            std::make_shared<EmptyRangeBase<EntityId>>());

        return;
    } else if (1.0 - ratio < EPSILON) {  // all in
        (*ranges)[prefix + "." + std::string(EntityIFORMName[VERTEX]) + "_BODY"].append(
            std::make_shared<ContinueRange<EntityId>>(GetIndexBox(0), 0));

        (*ranges)[prefix + "." + std::string(EntityIFORMName[EDGE]) + "_BODY"]
            .append(std::make_shared<ContinueRange<EntityId>>(GetIndexBox(1), 1))
            .append(std::make_shared<ContinueRange<EntityId>>(GetIndexBox(2), 2))
            .append(std::make_shared<ContinueRange<EntityId>>(GetIndexBox(4), 4));

        (*ranges)[prefix + "." + std::string(EntityIFORMName[FACE]) + "_BODY"]
            .append(std::make_shared<ContinueRange<EntityId>>(GetIndexBox(3), 3))
            .append(std::make_shared<ContinueRange<EntityId>>(GetIndexBox(5), 5))
            .append(std::make_shared<ContinueRange<EntityId>>(GetIndexBox(6), 6));

        (*ranges)[prefix + "." + std::string(EntityIFORMName[VOLUME]) + "_BODY"].append(
            std::make_shared<ContinueRange<EntityId>>(GetIndexBox(7), 7));
        return;
    }

    Field<StructuredMesh, int, VERTEX> vertex_tags{this};
    vertex_tags.Clear();

    index_tuple ib, ie;
    std::tie(ib, ie) = GetIndexBox(VERTEX);
    auto b = GetBox();

    for (index_type I = ib[0]; I < ie[0]; ++I)
        for (index_type J = ib[1]; J < ie[1]; ++J)
            for (index_type K = ib[2]; K < ie[2]; ++K) {
                auto x = point(EntityId{static_cast<int16_t>(I), static_cast<int16_t>(J), static_cast<int16_t>(K)});
                if (!g->CheckInside(x)) { vertex_tags[0](I, J, K) = 1; }
            }
    //

    /**
    *\verbatim
    *                ^s (dl)
    *               /
    *   (dz) t     /
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
    *       000-------------001---> r (dr)
    *
    *\endverbatim
    */

    auto VERTEX_body = std::make_shared<UnorderedRange<EntityId>>();
    auto EDGE_body = std::make_shared<UnorderedRange<EntityId>>();
    auto FACE_body = std::make_shared<UnorderedRange<EntityId>>();
    auto VOLUME_body = std::make_shared<UnorderedRange<EntityId>>();

    auto VERTEX_boundary = std::make_shared<UnorderedRange<EntityId>>();
    auto EDGE_PARA_boundary = std::make_shared<UnorderedRange<EntityId>>();
    auto FACE_PARA_boundary = std::make_shared<UnorderedRange<EntityId>>();
    auto EDGE_PERP_boundary = std::make_shared<UnorderedRange<EntityId>>();
    auto FACE_PERP_boundary = std::make_shared<UnorderedRange<EntityId>>();
    auto VOLUME_boundary = std::make_shared<UnorderedRange<EntityId>>();

    static const int b0 = 0b000;
    static const int b1 = 0b001;
    static const int b2 = 0b010;
    static const int b3 = 0b011;
    static const int b4 = 0b100;
    static const int b5 = 0b101;
    static const int b6 = 0b110;
    static const int b7 = 0b111;
    static const EntityId t0 = {0, 0, 0, 0b000};
    static const EntityId t1 = {0, 0, 0, 0b001};
    static const EntityId t2 = {0, 0, 0, 0b010};
    static const EntityId t3 = {0, 0, 0, 0b011};
    static const EntityId t4 = {0, 0, 0, 0b100};
    static const EntityId t5 = {0, 0, 0, 0b101};
    static const EntityId t6 = {0, 0, 0, 0b110};
    static const EntityId t7 = {0, 0, 0, 0b111};

    static const EntityId s0 = {0, 0, 0, 0};
    static const EntityId s1 = {1, 0, 0, 0};
    static const EntityId s2 = {0, 1, 0, 0};
    static const EntityId s3 = {1, 1, 0, 0};
    static const EntityId s4 = {0, 0, 1, 0};
    static const EntityId s5 = {1, 0, 1, 0};
    static const EntityId s6 = {0, 1, 1, 0};
    static const EntityId s7 = {1, 1, 1, 0};

    std::tie(ib, ie) = GetIndexBox(VOLUME);

#pragma omp parallel for
    for (index_type I = ib[0]; I < ie[0]; ++I)
        for (index_type J = ib[1]; J < ie[1]; ++J)
            for (index_type K = ib[2]; K < ie[2]; ++K) {
                EntityId s = {static_cast<int16_t>(I), static_cast<int16_t>(J), static_cast<int16_t>(K), 0};

                int volume_tags = ((vertex_tags[0](I + 0, J + 0, K + 0)) << 0) |  //
                                  ((vertex_tags[0](I + 1, J + 0, K + 0)) << 1) |  //
                                  ((vertex_tags[0](I + 0, J + 1, K + 0)) << 2) |  //
                                  ((vertex_tags[0](I + 1, J + 1, K + 0)) << 3) |  //
                                  ((vertex_tags[0](I + 0, J + 0, K + 1)) << 4) |  //
                                  ((vertex_tags[0](I + 1, J + 0, K + 1)) << 5) |  //
                                  ((vertex_tags[0](I + 0, J + 1, K + 1)) << 6) |  //
                                  ((vertex_tags[0](I + 1, J + 1, K + 1)) << 7);
                //
                if (volume_tags == 0) {
                    /**
                     *\verbatim
                     *                ^s (dl)
                     *               /
                     *   (dz) t     /
                     *        ^    /
                     *        |   6 --------------7
                     *        |  /|              /|
                     *        | / |             / |
                     *        |/  |            /  |
                     *        4 --|---------- 5   |
                     *        | m |           |   |
                     *        |   2 ----------|-- 3
                     *        |  /            |  /
                     *        | /             | /
                     *        |/              |/
                     *        0 ------------- 1---> r (dr)
                     *
                     *\endverbatim
                     */

                    VERTEX_body->Insert(s0 + s);
                    VERTEX_body->Insert(s1 + s);
                    VERTEX_body->Insert(s2 + s);
                    VERTEX_body->Insert(s3 + s);
                    VERTEX_body->Insert(s4 + s);
                    VERTEX_body->Insert(s5 + s);
                    VERTEX_body->Insert(s6 + s);
                    VERTEX_body->Insert(s7 + s);

                    EDGE_body->Insert((t1 | s0) + s);
                    EDGE_body->Insert((t1 | s2) + s);
                    EDGE_body->Insert((t1 | s4) + s);
                    EDGE_body->Insert((t1 | s6) + s);

                    EDGE_body->Insert((t2 | s0) + s);
                    EDGE_body->Insert((t2 | s1) + s);
                    EDGE_body->Insert((t2 | s4) + s);
                    EDGE_body->Insert((t2 | s5) + s);

                    EDGE_body->Insert((t4 | s0) + s);
                    EDGE_body->Insert((t4 | s1) + s);
                    EDGE_body->Insert((t4 | s2) + s);
                    EDGE_body->Insert((t4 | s3) + s);

                    FACE_body->Insert((t3 | s0) + s);
                    FACE_body->Insert((t5 | s0) + s);
                    FACE_body->Insert((t6 | s0) + s);

                    FACE_body->Insert((t6 | s1) + s);
                    FACE_body->Insert((t5 | s2) + s);
                    FACE_body->Insert((t3 | s4) + s);

                    VOLUME_body->Insert(t7 + s);

                } else if (volume_tags < 0b11111111) {
                    if ((volume_tags & b0) != 0) { VERTEX_boundary->Insert(s0 + s); }
                    if ((volume_tags & b1) != 0) { VERTEX_boundary->Insert(s1 + s); }
                    if ((volume_tags & b2) != 0) { VERTEX_boundary->Insert(s2 + s); }
                    if ((volume_tags & b3) != 0) { VERTEX_boundary->Insert(s3 + s); }
                    if ((volume_tags & b4) != 0) { VERTEX_boundary->Insert(s4 + s); }
                    if ((volume_tags & b5) != 0) { VERTEX_boundary->Insert(s5 + s); }
                    if ((volume_tags & b6) != 0) { VERTEX_boundary->Insert(s6 + s); }
                    if ((volume_tags & b7) != 0) { VERTEX_boundary->Insert(s7 + s); }

#define CHECK_TAG(_TAG_) if ((volume_tags & _TAG_) == _TAG_)
                    CHECK_TAG(0b00000011) { EDGE_PARA_boundary->Insert(t1 | s0 + s); }
                    CHECK_TAG(0b00001100) { EDGE_PARA_boundary->Insert(t1 | s2 + s); }
                    CHECK_TAG(0b00110000) { EDGE_PARA_boundary->Insert(t1 | s4 + s); }
                    CHECK_TAG(0b11000000) { EDGE_PARA_boundary->Insert(t1 | s6 + s); }
                    CHECK_TAG(0b00000101) { EDGE_PARA_boundary->Insert(t2 | s0 + s); }
                    CHECK_TAG(0b00001010) { EDGE_PARA_boundary->Insert(t2 | s1 + s); }
                    CHECK_TAG(0b01010000) { EDGE_PARA_boundary->Insert(t2 | s4 + s); }
                    CHECK_TAG(0b10100000) { EDGE_PARA_boundary->Insert(t2 | s5 + s); }
                    CHECK_TAG(0b00010001) { EDGE_PARA_boundary->Insert(t4 | s0 + s); }
                    CHECK_TAG(0b00100010) { EDGE_PARA_boundary->Insert(t4 | s1 + s); }
                    CHECK_TAG(0b01000100) { EDGE_PARA_boundary->Insert(t4 | s2 + s); }
                    CHECK_TAG(0b10001000) { EDGE_PARA_boundary->Insert(t4 | s3 + s); }

//                    CHECK_TAG(0b00000011) { FACE_PARA_boundary->Insert(t3 | s0 + s); }
//                    CHECK_TAG(0b00000011) { FACE_PARA_boundary->Insert(t5 | s0 + s); }
//                    CHECK_TAG(0b00000011) { FACE_PARA_boundary->Insert(t6 | s0 + s); }
//                    CHECK_TAG(0b00000011) { FACE_PARA_boundary->Insert(t6 | s1 + s); }
//                    CHECK_TAG(0b00000011) { FACE_PARA_boundary->Insert(t5 | s2 + s); }
//                    CHECK_TAG(0b00000011) { FACE_PARA_boundary->Insert(t3 | s4 + s); }
#undef CHECK_TAG
                    VOLUME_boundary->Insert(t7 + s);
                }
            }

    //    CHECK(prefix) << VERTEX_body->size() << "  " << g->GetBoundBox();

    (*ranges)[prefix + "." + std::string(EntityIFORMName[VERTEX]) + "_BODY"].append(VERTEX_body);
    (*ranges)[prefix + "." + std::string(EntityIFORMName[EDGE]) + "_BODY"].append(EDGE_body);
    (*ranges)[prefix + "." + std::string(EntityIFORMName[FACE]) + "_BODY"].append(FACE_body);
    (*ranges)[prefix + "." + std::string(EntityIFORMName[VOLUME]) + "_BODY"].append(VOLUME_body);
    (*ranges)[prefix + "." + std::string(EntityIFORMName[VERTEX]) + "_BOUNDARY"].append(VERTEX_boundary);
    (*ranges)[prefix + "." + std::string(EntityIFORMName[EDGE]) + "_PARA_BOUNDARY"].append(EDGE_PARA_boundary);
    (*ranges)[prefix + "." + std::string(EntityIFORMName[FACE]) + "_PARA_BOUNDARY"].append(FACE_PARA_boundary);
    (*ranges)[prefix + "." + std::string(EntityIFORMName[EDGE]) + "_PERP_BOUNDARY"].append(EDGE_PERP_boundary);
    (*ranges)[prefix + "." + std::string(EntityIFORMName[FACE]) + "_PERP_BOUNDARY"].append(FACE_PERP_boundary);
    (*ranges)[prefix + "." + std::string(EntityIFORMName[VOLUME]) + "_BOUNDARY"].append(VOLUME_boundary);

    //    CHECK(VOLUME_body->size());
}

}  // namespace mesh{
}  // namespace simpla{
