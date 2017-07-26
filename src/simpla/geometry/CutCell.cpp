//
// Created by salmon on 17-7-26.
//

#include "CutCell.h"

#include <BRepIntCurveSurface_Inter.hxx>

#include "BoxUtilities.h"
#include "Chart.h"
#include "GeoObject.h"
#include "occ/OCCShape.h"
namespace simpla {
namespace geometry {

void CutCell(Chart *chart, index_box_type const &m_idx_box, point_type const &r, geometry::GeoObject const *g,
             Range<EntityId> body_ranges[4], std::map<EntityId, Real> cut_cell[4]) {
    box_type m_box{chart->global_coordinates(std::get<0>(m_idx_box)),
                   chart->global_coordinates(std::get<1>(m_idx_box))};

    auto g_box = g->BoundingBox();

    index_box_type g_idx_box{std::get<0>(chart->invert_local_coordinates(std::get<0>(g_box))),
                             std::get<0>(chart->invert_local_coordinates(std::get<1>(g_box)))};

    auto const &scale = chart->GetScale();
    Real tol = std::sqrt(dot(scale, scale) * 0.25);

    BRepIntCurveSurface_Inter m_inter_;

    m_inter_.Load(*geometry::occ_cast<TopoDS_Shape>(*g), tol);

    Array<int, ZSFC<3>> vertex_tags(nullptr, m_idx_box);
    vertex_tags.Clear();
    std::map<EntityId, Real> m_edge_fraction;

    for (int dir = 0; dir < 3; ++dir) {
        std::vector<Real> res;
        res.clear();
        index_tuple lo{0, 0, 0}, hi{0, 0, 0};
        std::tie(lo, hi) = m_idx_box;
        lo[dir] = std::min(std::get<0>(g_idx_box)[dir], std::get<0>(m_idx_box)[dir]);
        hi[dir] = lo[dir] + 1;
        //        hi[dir] = std::max(std::get<1>(g_idx_box)[dir], std::get<1>(m_idx_box)[dir]);

        for (index_type i = lo[0]; i < hi[0]; ++i)
            for (index_type j = lo[1]; j < hi[1]; ++j)
                for (index_type k = lo[0]; k < hi[2]; ++k) {
                    point_type x0 = chart->local_coordinates(index_tuple{i, j, k}, 0b0);
                    m_inter_.Init(Handle(Geom_Curve)(geometry::occ_cast<Geom_Curve>(chart->GetAxisCurve(x0, dir))));
                    if (m_inter_.Transition() != IntCurveSurface_In) { continue; }
                    bool in_box = false;
                    bool in_side = false;

                    index_tuple idx{i, j, k};

                    index_type s0 = lo[dir], s1 = lo[dir];

                    for (; m_inter_.More(); m_inter_.Next()) {
                        point_type x{m_inter_.Pnt().X(), m_inter_.Pnt().Y(), m_inter_.Pnt().Z()};
                        in_side = !in_side;

                        auto l_coor = chart->invert_local_coordinates(x);

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
                            ASSERT(s1 > s0);
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
    //        TagCutVertices(chart, &vertex_tags, dynamic_cast<geometry::GeoObjectOCC const *>(g));
    //    } else {
    //        auto const *chart = chart->GetChart();
    //        vertex_tags = [&](index_type x, index_type y, index_type z) {
    //            return g->CheckInside(
    //                       chart->xyz(point_type{static_cast<Real>(x), static_cast<Real>(y), static_cast<Real>(z)}))
    //                       ? 1
    //                       : 0;
    //        };
    //    }
    //
    UpdateRanges(chart, prefix, vertex_tags);
}

}  //    namespace geometry{

}  // namespace simpla{