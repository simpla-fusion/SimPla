//
// Created by salmon on 17-7-26.
//

#include "CutCell.h"

#include <BRepIntCurveSurface_Inter.hxx>

#include <simpla/algebra/nTuple.ext.h>

#include "BoxUtilities.h"
#include "Chart.h"
#include "GeoObject.h"
#include "occ/OCCShape.h"
namespace simpla {
namespace geometry {

void CutCell(Chart *chart, index_box_type m_idx_box, GeoObject const *g, Range<EntityId> *body_ranges,
             Range<EntityId> *boundary_ranges, std::map<EntityId, Real> *cut_cell, Array<Real> *edge_fraction,
             Array<Real> *vertex_tags) {
    auto const &scale = chart->GetScale();
    Real tol = std::sqrt(dot(scale, scale) * 0.01);
    std::get<1>(m_idx_box) += 1;
    BRepIntCurveSurface_Inter m_inter_;

    m_inter_.Load(*geometry::occ_cast<TopoDS_Shape>(*g), 0.0001);

    //    Array<int, ZSFC<3>> vertex_tags(nullptr, m_idx_box);
    //    vertex_tags.Clear();
    //    std::map<EntityId, Real> m_edge_fraction;

    for (int dir = 2; dir < 3; ++dir) {
        index_tuple lo{0, 0, 0}, hi{0, 0, 0};
        std::tie(lo, hi) = m_idx_box;
        hi[dir] = lo[dir] + 1;
        size_type count = 0;
        for (index_type i = lo[0]; i < hi[0]; ++i)
            for (index_type j = lo[1]; j < hi[1]; ++j)
                for (index_type k = lo[2]; k < hi[2]; ++k) {
                    Handle(Geom_Curve) c = geometry::detail::OCCCast<Geom_Curve, Curve>::eval(
                        *chart->GetAxisCurve(index_tuple{i, j, k}, dir));
                    m_inter_.Init(c);

                    index_type s0 = lo[dir], s1 = lo[dir];

                    bool is_inside = false;
                    for (; m_inter_.More(); m_inter_.Next()) {
                        is_inside = !is_inside;
                        point_type x{m_inter_.Pnt().X(), m_inter_.Pnt().Y(), m_inter_.Pnt().Z()};

                        ++count;

                        index_tuple idx{0, 0, 0};
                        point_type r{0, 0, 0};
                        std::tie(idx, r) = chart->invert_global_coordinates(x);

                        // std::cout << index_tuple{i, j, k} << "~" << idx << "~" << r <<
                        // std::endl;
                        // vertex_tags->Set(count, idx);
                        // std::cout << "DIR:" << dir << "\t" << m_idx_box << "\t" <<
                        // index_tuple{i, j, k} << "\t" << idx;
                        // if (!(CheckInSide(m_idx_box, idx))) {
                        //    std::cout << std::endl;
                        //    continue;
                        // } else {
                        //    std::cout << "\t" << (x) << "\t" << chart->inv_map(x) <<
                        //    std::endl;
                        // }
                        edge_fraction[dir].Set(r[dir], idx);
                        //                        vertex_tags->Set(1, idx);
                        //                        idx[(dir + 1) % 3] -= 1;
                        //                        vertex_tags->Set(1, idx);
                        //                        idx[(dir + 2) % 3] -= 1;
                        //                        vertex_tags->Set(1, idx);
                        //                        idx[(dir + 1) % 3] += 1;
                        //                        vertex_tags->Set(1, idx);
                        //                        index_tuple id{i, j, k};
                        //                        id[dir] = std::get<0>(l_coor)[dir];
                        //                        vertex_tags[0].Set(dir + 1, id);
                        //                        id[(dir + 1) % 3] = idx[(dir + 1) % 3] - 1;
                        //                        vertex_tags[0].Set(dir + 1, id);
                        //                        id[(dir + 2) % 3] = idx[(dir + 2) % 3] - 1;
                        //                        vertex_tags[0].Set(dir + 1, id);
                        //                        id[(dir + 1) % 3] = idx[(dir + 1) % 3];
                        //                        vertex_tags[0].Set(dir + 1, id);
                        //                        if (m_inter_.State() == TopAbs_IN) {
                        //                            s0 = std::max(std::get<0>(l_coor)[dir],
                        //                            std::get<0>(m_idx_box)[dir]);
                        //                        }
                        //
                        //                        if (x[dir] < std::get<0>(m_box)[dir]) { continue; }
                        //
                        //                        EntityId q;
                        //                        q.x = static_cast<int16_t>(std::get<0>(l_coor)[0]);
                        //                        q.y = static_cast<int16_t>(std::get<0>(l_coor)[1]);
                        //                        q.z = static_cast<int16_t>(std::get<0>(l_coor)[2]);
                        //                        q.w =
                        //                        static_cast<int16_t>(EntityIdCoder::m_sub_index_to_id_[EDGE][dir]);
                        //                        index_tuple idx{i, j, k};
                        //                        idx[dir] = std::get<0>(l_coor)[dir];
                        //                        edge_fraction[dir].Set(std::get<1>(l_coor)[dir], idx);
                        //

                        if (is_inside) {
                            s0 = idx[dir];
                            vertex_tags[0].Set(2, idx);
                        } else {
                            s1 = idx[dir];

                            vertex_tags[0].Set(2, idx);
                            s0 = std::max(std::get<0>(m_idx_box)[dir], s0);
                            s1 = std::min(std::get<1>(m_idx_box)[dir], s1);

                            for (index_type s = s0 + 1; s <= s1; ++s) {
                                index_tuple id = idx;
                                id[dir] = s;
                                vertex_tags[0].Set(1, id);
                            }
                        }
                        //
                        //                        VERBOSE << "s0:" << s0 << " s1:" << s1 << std::endl;
                        //
                        //                        if (x[dir] > std::get<1>(m_idx_box)[dir]) { break; }
                    }
                }

    }
}

}  //    namespace geometry{

}  // namespace simpla{