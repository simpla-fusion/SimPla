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

void CutCell(Chart *chart, index_box_type const &m_idx_box, point_type const &r, GeoObject const *g,
             Range<EntityId> body_ranges[4], Range<EntityId> boundary_ranges[4], std::map<EntityId, Real> cut_cell[4],
             Array<Real> edge_fraction[3], Array<Real> *cell_tags) {
    auto const &scale = chart->GetScale();
    Real tol = std::sqrt(dot(scale, scale) * 0.01);

    BRepIntCurveSurface_Inter m_inter_;

    m_inter_.Load(*geometry::occ_cast<TopoDS_Shape>(*g), 0.0001);

    //    Array<int, ZSFC<3>> vertex_tags(nullptr, m_idx_box);
    //    vertex_tags.Clear();
    //    std::map<EntityId, Real> m_edge_fraction;

    for (int dir = 0; dir < 3; ++dir) {
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
                    for (; m_inter_.More(); m_inter_.Next()) {
                        point_type x{m_inter_.Pnt().X(), m_inter_.Pnt().Y(), m_inter_.Pnt().Z()};

                        ++count;

                        index_tuple idx{0, 0, 0};
                        point_type r{0, 0, 0};
                        std::tie(idx, r) = chart->invert_global_coordinates(x);

                        //                        cell_tags->Set(count, idx);
                        std::cout << "DIR:" << dir << "\t" << m_idx_box << "\t" << index_tuple{i, j, k} << "\t" << idx;
                        if (!(CheckInSide(m_idx_box, idx))) {
                            std::cout << std::endl;
//                            continue;
                        } else {
                            std::cout << "\t" << (x) << "\t" << chart->inv_map(x) << std::endl;
                        }
                        //                        edge_fraction[dir].Set(r[dir], idx);
                        cell_tags->Set(count, idx);
                        idx[(dir + 1) % 3] -= 1;
                        cell_tags->Set(count, idx);
                        idx[(dir + 2) % 3] -= 1;
                        cell_tags->Set(count, idx);
                        idx[(dir + 1) % 3] += 1;
                        cell_tags->Set(count, idx);

                        //                        index_tuple id{i, j, k};
                        //                        id[dir] = std::get<0>(l_coor)[dir];
                        //                        cell_tags[0].Set(dir + 1, id);
                        //                        id[(dir + 1) % 3] = idx[(dir + 1) % 3] - 1;
                        //                        cell_tags[0].Set(dir + 1, id);
                        //                        id[(dir + 2) % 3] = idx[(dir + 2) % 3] - 1;
                        //                        cell_tags[0].Set(dir + 1, id);
                        //                        id[(dir + 1) % 3] = idx[(dir + 1) % 3];
                        //                        cell_tags[0].Set(dir + 1, id);
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
                        //                        if (m_inter_.State() == TopAbs_IN) {
                        //                            edge_fraction[dir].Set(-std::get<1>(l_coor)[dir], idx);
                        //                        } else {
                        //                            s1 = std::min(std::get<0>(l_coor)[dir],
                        //                            std::get<1>(m_idx_box)[dir]);
                        //                            ASSERT(s1 > s0);
                        //                            for (index_type s = s0; s < s1; ++s) {
                        //                                idx[dir] = s;
                        //                                m_volume_tag_[0](idx) = 1;
                        //                            }
                        //                        }
                        //
                        //                        VERBOSE << "s0:" << s0 << " s1:" << s1 << std::endl;
                        //
                        //                        if (x[dir] > std::get<1>(m_idx_box)[dir]) { break; }
                    }
                }

        std::cout << "Count :" << count << " Dir:" << dir << " box:" << m_idx_box << std::endl;
    }
}

}  //    namespace geometry{

}  // namespace simpla{