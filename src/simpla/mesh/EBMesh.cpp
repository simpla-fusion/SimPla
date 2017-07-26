//
// Created by salmon on 17-7-18.
//

#include "EBMesh.h"
#include <BRepIntCurveSurface_Inter.hxx>

#include "simpla/engine/Mesh.h"
#include "simpla/geometry/BoxUtilities.h"
#include "simpla/geometry/GeoObject.h"
#include "simpla/geometry/occ/OCCShape.h"

namespace simpla {
namespace mesh {
namespace detail {
struct EntityIdHasher {
    size_t operator()(const EntityId &key) const { return static_cast<size_t>(key.v); }
};

std::shared_ptr<UnorderedRange<EntityId>> make_range(tbb::concurrent_unordered_set<EntityId, EntityIdHasher> const &t) {
    auto res = std::make_shared<UnorderedRange<EntityId>>();
    res->reset(t.size(), nullptr);
    std::copy(t.begin(), t.end(), res->get());
    return res;
}

void UpdateRanges(engine::MeshBase *m_host_, std::string const &prefix, Array<int, ZSFC<3>> const &vertex_tags);

void CreateEBMesh(engine::MeshBase *p_mesh, std::string const &prefix, geometry::GeoObject const *g) {
    auto m_box = p_mesh->BoundingBox(0b0);
    index_box_type m_idx_box = p_mesh->IndexBox(0b0);

    auto g_box = g->BoundingBox();
    index_box_type g_idx_box{std::get<0>(p_mesh->GetChart()->invert_local_coordinates(std::get<0>(g_box))),
                             std::get<0>(p_mesh->GetChart()->invert_local_coordinates(std::get<1>(g_box)))};

    auto scale = p_mesh->GetChart()->GetScale();
    Real tol = std::sqrt(dot(scale, scale) * 0.25);

    BRepIntCurveSurface_Inter m_inter_;

    m_inter_.Load(*geometry::occ_cast<TopoDS_Shape>(*g), tol);

    Array<int, ZSFC<3>> vertex_tags(nullptr, p_mesh->IndexBox(0b0));
    vertex_tags.Clear();
    std::map<EntityId, Real> m_edge_fraction;

    for (int dir = 0; dir < 3; ++dir) {
        std::vector<Real> res;
        res.clear();
        index_tuple lo, hi;
        std::tie(lo, hi) = m_idx_box;
        lo[dir] = std::min(std::get<0>(g_idx_box)[dir], std::get<0>(m_idx_box)[dir]);
        hi[dir] = lo[dir] + 1;
        //        hi[dir] = std::max(std::get<1>(g_idx_box)[dir], std::get<1>(m_idx_box)[dir]);

        for (index_type i = lo[0]; i < hi[0]; ++i)
            for (index_type j = lo[1]; j < hi[1]; ++j)
                for (index_type k = lo[0]; k < hi[2]; ++k) {
                    point_type x0 = p_mesh->GetChart()->local_coordinates(index_tuple{i, j, k}, 0b0);
                    m_inter_.Init(
                        Handle(Geom_Curve)(geometry::occ_cast<Geom_Curve>(p_mesh->GetChart()->GetAxisCurve(x0, dir))));
                    if (m_inter_.Transition() != IntCurveSurface_In) { continue; }
                    bool in_box = false;
                    bool in_side = false;

                    index_tuple idx{i, j, k};

                    index_type s0 = lo[dir], s1 = lo[dir];

                    for (; m_inter_.More(); m_inter_.Next()) {
                        point_type x{m_inter_.Pnt().X(), m_inter_.Pnt().Y(), m_inter_.Pnt().Z()};
                        in_side = !in_side;

                        auto l_coor = p_mesh->GetChart()->invert_local_coordinates(x);

                        if (in_side) { s0 = std::max(std::get<0>(l_coor)[dir], std::get<0>(m_idx_box)[dir]); }

                        if (x[dir] < std::get<0>(m_box)[dir]) { continue; }

                        EntityId q;
                        q.x = static_cast<int16_t>(std::get<0>(l_coor)[0]);
                        q.y = static_cast<int16_t>(std::get<0>(l_coor)[1]);
                        q.z = static_cast<int16_t>(std::get<0>(l_coor)[2]);
                        q.w = static_cast<int16_t>(EntityIdCoder::m_sub_index_to_id_[EDGE][dir]);

                        m_edge_fraction[q] = std::get<1>(l_coor)[dir];

                        if (in_side) {
                            m_edge_fraction[q] *= -1;
                        } else {
                            s1 = std::min(std::get<0>(l_coor)[dir], std::get<1>(m_idx_box)[dir]);
                            for (index_type s = s0; s < s1; ++s) {
                                idx[dir] = s;
                                vertex_tags(idx) = 1;
                            }
                        }

                        if (x[dir] > std::get<1>(m_idx_box)[dir]) { break; }
                    }
                }
    }
    //    Array<int, ZSFC<3>> vertex_tags{nullptr, idx_box};
    //    if (g->isA(typeid(geometry::GeoObjectOCC))) {
    //        TagCutVertices(p_mesh, &vertex_tags, dynamic_cast<geometry::GeoObjectOCC const *>(g));
    //    } else {
    //        auto const *chart = p_mesh->GetChart();
    //        vertex_tags = [&](index_type x, index_type y, index_type z) {
    //            return g->CheckInside(
    //                       chart->xyz(point_type{static_cast<Real>(x), static_cast<Real>(y), static_cast<Real>(z)}))
    //                       ? 1
    //                       : 0;
    //        };
    //    }
    //
    UpdateRanges(p_mesh, prefix, vertex_tags);
}
void UpdateRanges(engine::MeshBase *m_host_, std::string const &prefix, Array<int, ZSFC<3>> const &vertex_tags) {
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

    //    Array<int, ZSFC<3>> volume_tags{nullptr, m_host_->IndexBox(0b111)};
    //
    //    volume_tags = ((vertex_tags(IdxShift{0, 0, 0})) << 0) |  //
    //                  ((vertex_tags(IdxShift{1, 0, 0})) << 1) |  //
    //                  ((vertex_tags(IdxShift{0, 1, 0})) << 2) |  //
    //                  ((vertex_tags(IdxShift{1, 1, 0})) << 3) |  //
    //                  ((vertex_tags(IdxShift{0, 0, 1})) << 4) |  //
    //                  ((vertex_tags(IdxShift{1, 0, 1})) << 5) |  //
    //                  ((vertex_tags(IdxShift{0, 1, 1})) << 6) |  //
    //                  ((vertex_tags(IdxShift{1, 1, 1})) << 7);

    tbb::concurrent_unordered_set<EntityId, detail::EntityIdHasher> VERTEX_body;
    tbb::concurrent_unordered_set<EntityId, detail::EntityIdHasher> EDGE_body;
    tbb::concurrent_unordered_set<EntityId, detail::EntityIdHasher> FACE_body;
    tbb::concurrent_unordered_set<EntityId, detail::EntityIdHasher> VOLUME_body;

    tbb::concurrent_unordered_set<EntityId, detail::EntityIdHasher> VERTEX_boundary;
    tbb::concurrent_unordered_set<EntityId, detail::EntityIdHasher> EDGE_boundary;
    tbb::concurrent_unordered_set<EntityId, detail::EntityIdHasher> FACE_boundary;
    tbb::concurrent_unordered_set<EntityId, detail::EntityIdHasher> EDGE_CUT_boundary;
    tbb::concurrent_unordered_set<EntityId, detail::EntityIdHasher> FACE_CUT_boundary;
    tbb::concurrent_unordered_set<EntityId, detail::EntityIdHasher> VOLUME_boundary;

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

    ZSFC<3>(geometry::Expand(m_host_->IndexBox(0b111), index_tuple{1, 1, 1}))
        .Foreach(  //
            [&](auto I, auto J, auto K) {

                EntityId s = {static_cast<int16_t>(I), static_cast<int16_t>(J), static_cast<int16_t>(K), 0};

                int tag = ((vertex_tags(I + 0, J + 0, K + 0)) << 0) |  //
                          ((vertex_tags(I + 1, J + 0, K + 0)) << 1) |  //
                          ((vertex_tags(I + 0, J + 1, K + 0)) << 2) |  //
                          ((vertex_tags(I + 1, J + 1, K + 0)) << 3) |  //
                          ((vertex_tags(I + 0, J + 0, K + 1)) << 4) |  //
                          ((vertex_tags(I + 1, J + 0, K + 1)) << 5) |  //
                          ((vertex_tags(I + 0, J + 1, K + 1)) << 6) |  //
                          ((vertex_tags(I + 1, J + 1, K + 1)) << 7);

                //
                if (tag == 0b11111111) {
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

                    VERTEX_body.insert(s0 + s);
                    VERTEX_body.insert(s1 + s);
                    VERTEX_body.insert(s2 + s);
                    VERTEX_body.insert(s3 + s);
                    VERTEX_body.insert(s4 + s);
                    VERTEX_body.insert(s5 + s);
                    VERTEX_body.insert(s6 + s);
                    VERTEX_body.insert(s7 + s);

                    EDGE_body.insert((t1 | s0) + s);
                    EDGE_body.insert((t1 | s2) + s);
                    EDGE_body.insert((t1 | s4) + s);
                    EDGE_body.insert((t1 | s6) + s);

                    EDGE_body.insert((t2 | s0) + s);
                    EDGE_body.insert((t2 | s1) + s);
                    EDGE_body.insert((t2 | s4) + s);
                    EDGE_body.insert((t2 | s5) + s);

                    EDGE_body.insert((t4 | s0) + s);
                    EDGE_body.insert((t4 | s1) + s);
                    EDGE_body.insert((t4 | s2) + s);
                    EDGE_body.insert((t4 | s3) + s);

                    FACE_body.insert((t3 | s0) + s);
                    FACE_body.insert((t5 | s0) + s);
                    FACE_body.insert((t6 | s0) + s);

                    FACE_body.insert((t6 | s1) + s);
                    FACE_body.insert((t5 | s2) + s);
                    FACE_body.insert((t3 | s4) + s);

                    VOLUME_body.insert(t7 + s);

                } else if (tag > 0 && tag < 0b11111111) {
                    if ((tag & b0) != 0) { VERTEX_boundary.insert(s0 + s); }
                    if ((tag & b1) != 0) { VERTEX_boundary.insert(s1 + s); }
                    if ((tag & b2) != 0) { VERTEX_boundary.insert(s2 + s); }
                    if ((tag & b3) != 0) { VERTEX_boundary.insert(s3 + s); }
                    if ((tag & b4) != 0) { VERTEX_boundary.insert(s4 + s); }
                    if ((tag & b5) != 0) { VERTEX_boundary.insert(s5 + s); }
                    if ((tag & b6) != 0) { VERTEX_boundary.insert(s6 + s); }
                    if ((tag & b7) != 0) { VERTEX_boundary.insert(s7 + s); }

#define CHECK_TAG(_TAG_) if ((tag & _TAG_) == _TAG_)
                    CHECK_TAG(0b00000011) { EDGE_boundary.insert(t1 | s0 + s); }
                    CHECK_TAG(0b00001100) { EDGE_boundary.insert(t1 | s2 + s); }
                    CHECK_TAG(0b00110000) { EDGE_boundary.insert(t1 | s4 + s); }
                    CHECK_TAG(0b11000000) { EDGE_boundary.insert(t1 | s6 + s); }
                    CHECK_TAG(0b00000101) { EDGE_boundary.insert(t2 | s0 + s); }
                    CHECK_TAG(0b00001010) { EDGE_boundary.insert(t2 | s1 + s); }
                    CHECK_TAG(0b01010000) { EDGE_boundary.insert(t2 | s4 + s); }
                    CHECK_TAG(0b10100000) { EDGE_boundary.insert(t2 | s5 + s); }
                    CHECK_TAG(0b00010001) { EDGE_boundary.insert(t4 | s0 + s); }
                    CHECK_TAG(0b00100010) { EDGE_boundary.insert(t4 | s1 + s); }
                    CHECK_TAG(0b01000100) { EDGE_boundary.insert(t4 | s2 + s); }
                    CHECK_TAG(0b10001000) { EDGE_boundary.insert(t4 | s3 + s); }

                    CHECK_TAG(0b00000011) { FACE_boundary.insert(t3 | s0 + s); }
                    CHECK_TAG(0b00000011) { FACE_boundary.insert(t5 | s0 + s); }
                    CHECK_TAG(0b00000011) { FACE_boundary.insert(t6 | s0 + s); }
                    CHECK_TAG(0b00000011) { FACE_boundary.insert(t6 | s1 + s); }
                    CHECK_TAG(0b00000011) { FACE_boundary.insert(t5 | s2 + s); }
                    CHECK_TAG(0b00000011) { FACE_boundary.insert(t3 | s4 + s); }
#undef CHECK_TAG
                    VOLUME_boundary.insert(t7 + s);
                }
            });

    CHECK(VOLUME_body.size());

    m_host_->GetRange(prefix + "_BODY_0").append(make_range(VERTEX_body));
    m_host_->GetRange(prefix + "_BODY_1").append(make_range(EDGE_body));
    m_host_->GetRange(prefix + "_BODY_2").append(make_range(FACE_body));
    m_host_->GetRange(prefix + "_BODY_3").append(make_range(VOLUME_body));

    m_host_->GetRange(prefix + "_BOUNDARY_0").append(make_range(VERTEX_boundary));
    m_host_->GetRange(prefix + "_BOUNDARY_3").append(make_range(VOLUME_boundary));

    m_host_->GetRange(prefix + "_BOUNDARY_1").append(make_range(EDGE_boundary));
    m_host_->GetRange(prefix + "_BOUNDARY_2").append(make_range(FACE_boundary));

    m_host_->GetRange(prefix + "_CUT_BOUNDARY_1").append(make_range(EDGE_CUT_boundary));
    m_host_->GetRange(prefix + "_CUT_BOUNDARY_2").append(make_range(FACE_CUT_boundary));
}

}  // namespace detail {
}  // namespace mesh {
}  // namespace simpla{
