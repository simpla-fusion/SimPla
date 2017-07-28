//
// Created by salmon on 17-7-26.
//

#include "CutCell.h"

#include <BRepIntCurveSurface_Inter.hxx>

#include <simpla/algebra/nTuple.ext.h>
#include <BRepPrimAPI_MakeBox.hxx>

#include "BoxUtilities.h"
#include "Chart.h"
#include "GeoObject.h"
#include "occ/OCCShape.h"
namespace simpla {
namespace geometry {

void CutCell(Chart *chart, index_box_type m_idx_box, GeoObject const *g, Array<Real> *vertex_tags) {
    auto const &scale = chart->GetScale();
    Real tol = std::sqrt(dot(scale, scale) * 0.01);
    std::get<1>(m_idx_box) += 1;
    auto bnd_box = g->BoundingBox();
    gp_Pnt xlo{std::get<0>(bnd_box)[0], std::get<0>(bnd_box)[1], std::get<0>(bnd_box)[2]};
    gp_Pnt xhi{std::get<1>(bnd_box)[0], std::get<1>(bnd_box)[1], std::get<1>(bnd_box)[2]};
    CHECK(bnd_box);
    BRepPrimAPI_MakeBox makeBox(xlo, xhi);
    makeBox.Build();
    auto box = makeBox.Shell();
    BRepIntCurveSurface_Inter m_box_inter_;
    m_box_inter_.Load(box, 0.0001);

    BRepIntCurveSurface_Inter m_body_inter_;
    m_body_inter_.Load(*geometry::occ_cast<TopoDS_Shape>(*g), 0.0001);

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
                    point_type x0 = chart->global_coordinates(i, j, k, 0b0);

                    // start point is on the bounding box
                    {
                        index_tuple idx{i, j, k};

                        index_type s0 = idx[dir];
                        Handle(Geom_Curve) c =
                            geometry::detail::OCCCast<Geom_Curve, Curve>::eval(*chart->GetAxisCurve(x0, dir));

                        m_box_inter_.Init(c);

                        // if curve do not intersect with bounding box then continue to next curve
                        if (!m_box_inter_.More()) { continue; }

                        bool is_first = true;
                        // search min intersection point
                        while (m_box_inter_.More()) {
                            index_tuple i1{0, 0, 0};
                            point_type x1{m_box_inter_.Pnt().X(), m_box_inter_.Pnt().Y(), m_box_inter_.Pnt().Z()};
                            std::tie(i1, std::ignore) = chart->invert_global_coordinates(x1);

                            if (is_first || i1[dir] < s0) {
                                s0 = i1[dir];
                                x0 = x1;
                                is_first = false;
                            }
                            m_box_inter_.Next();
                        }
                    }

                    // 2. new curve start from bounding box
                    Handle(Geom_Curve) c =
                        geometry::detail::OCCCast<Geom_Curve, Curve>::eval(*chart->GetAxisCurve(x0, dir));
                    m_body_inter_.Init(c);

                    if (m_body_inter_.More()) { std::cout << "==============" << std::endl; }
                    bool is_first = true;
                    index_type s0 = 0;
                    while (m_body_inter_.More()) {
                        ++count;

                        point_type x0{m_body_inter_.Pnt().X(), m_body_inter_.Pnt().Y(), m_body_inter_.Pnt().Z()};
                        std::cout << (x0) << " status:" << m_body_inter_.Transition() << std::endl;
                        index_tuple i0{0, 0, 0};
                        point_type r0{0, 0, 0};
                        std::tie(i0, r0) = chart->invert_global_coordinates(x0);

                        if (!is_first && m_body_inter_.Transition() == IntCurveSurface_Out) {
                            for (index_type s = s0; s <= i0[dir]; ++s) {
                                index_tuple id{i, j, k};
                                id[dir] = s;
                                vertex_tags[0].Set(1, id);
                            }
                        }
                        s0 = i0[dir];
                        m_body_inter_.Next();
                        is_first = false;
                    }

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
                    //   edge_fraction[dir].Set(r[dir], idx);
                    //   vertex_tags->Set(1, idx);
                    //   idx[(dir + 1) % 3] -= 1;
                    //   vertex_tags->Set(1, idx);
                    //   idx[(dir + 2) % 3] -= 1;
                    //   vertex_tags->Set(1, idx);
                    //   idx[(dir + 1) % 3] += 1;
                    //   vertex_tags->Set(1, idx);
                    //   index_tuple id{i, j, k};
                    //   id[dir] = std::get<0>(l_coor)[dir];
                    //   vertex_tags[0].Set(dir + 1, id);
                    //   id[(dir + 1) % 3] = idx[(dir + 1) % 3] - 1;
                    //   vertex_tags[0].Set(dir + 1, id);
                    //   id[(dir + 2) % 3] = idx[(dir + 2) % 3] - 1;
                    //   vertex_tags[0].Set(dir + 1, id);
                    //   id[(dir + 1) % 3] = idx[(dir + 1) % 3];
                    //   vertex_tags[0].Set(dir + 1, id);
                    //   if (m_body_inter_.State() == TopAbs_IN) {
                    //       s0 = std::max(std::get<0>(l_coor)[dir],
                    //       std::get<0>(m_idx_box)[dir]);
                    //   }
                    //
                    //   if (x[dir] < std::get<0>(m_box)[dir]) { continue; }
                    //
                    //   EntityId q;
                    //   q.x = static_cast<int16_t>(std::get<0>(l_coor)[0]);
                    //   q.y = static_cast<int16_t>(std::get<0>(l_coor)[1]);
                    //   q.z = static_cast<int16_t>(std::get<0>(l_coor)[2]);
                    //   q.w =
                    //   static_cast<int16_t>(EntityIdCoder::m_sub_index_to_id_[EDGE][dir]);
                    //   index_tuple idx{i, j, k};
                    //   idx[dir] = std::get<0>(l_coor)[dir];
                    //   edge_fraction[dir].Set(std::get<1>(l_coor)[dir], idx);
                    //

                    //                        VERBOSE << "s0:" << s0 << " s1:" << s1 << std::endl;
                    //                        if (x[dir] > std::get<1>(m_idx_box)[dir]) { break; }
                }
    }
}

}  //    namespace geometry{

}  // namespace simpla{