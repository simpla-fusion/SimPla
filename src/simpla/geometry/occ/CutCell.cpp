//
// Created by salmon on 17-7-26.
//

#include "../CutCell.h"

#include <simpla/algebra/nTuple.ext.h>
#include <BRepIntCurveSurface_Inter.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <GeomAdaptor_Curve.hxx>
#include <Geom_Curve.hxx>
#include <TopoDS_Shape.hxx>

#include "../BoxUtilities.h"
#include "../Chart.h"
#include "../GeoObject.h"

#include "OCCShape.h"
namespace simpla {
namespace geometry {

void CutCell(std::shared_ptr<Chart> const &chart, index_box_type m_idx_box, const std::shared_ptr<GeoObject> &g,
             Array<Real> *vertex_tags) {
    auto const &scale = chart->GetScale();
    Real tol = std::sqrt(dot(scale, scale) * 0.01);
    std::get<1>(m_idx_box) += 1;
    box_type bnd_box = g->GetBoundingBox();
    vector_type length = std::get<1>(bnd_box) - std::get<0>(bnd_box);
    std::get<0>(bnd_box) -= 0.03 * length;
    std::get<1>(bnd_box) += 0.03 * length;

    gp_Pnt xlo{std::get<0>(bnd_box)[0], std::get<0>(bnd_box)[1], std::get<0>(bnd_box)[2]};
    gp_Pnt xhi{std::get<1>(bnd_box)[0], std::get<1>(bnd_box)[1], std::get<1>(bnd_box)[2]};

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
        for (index_type i = lo[0]; i < hi[0]; ++i)
            for (index_type j = lo[1]; j < hi[1]; ++j)
                for (index_type k = lo[2]; k < hi[2]; ++k) {
                    //                    point_type x_begin = chart->global_coordinates(i, j, k, 0b0);
                    // start point is on the bounding box
                    //                    {
                    //                        index_tuple idx{i, j, k};
                    //
                    //                        index_type s0 = idx[dir];
                    //                        Handle(Geom_Curve) c =
                    //                            geometry::detail::OCCCast<Geom_Curve,
                    //                            Curve>::eval(*chart->GetAxisCurve(x_begin, dir));
                    //
                    //                        m_box_inter_.Init(c);
                    //
                    //                        // if curve do not intersect with bounding box then continue to next curve
                    //                        if (!m_box_inter_.More()) { continue; }
                    //
                    //                        bool is_first = true;
                    //                        // search min intersection point
                    //                        while (m_box_inter_.More()) {
                    //                            index_tuple i1{0, 0, 0};
                    //                            point_type x1{m_box_inter_.Pnt().X(), m_box_inter_.Pnt().Y(),
                    //                            m_box_inter_.Pnt().Z()};
                    //                            std::tie(i1, std::ignore) = chart->invert_global_coordinates(x1);
                    //
                    //                            if (is_first || i1[dir] < s0) {
                    //                                s0 = i1[dir];
                    //                                x_begin = x1;
                    //                                is_first = false;
                    //                            }
                    //                            m_box_inter_.Next();
                    //                        }
                    //                    }

                    point_type x_begin = chart->global_coordinates(0b0, i, j, k);
                    Handle(Geom_Curve) c =
                        geometry::detail::OCCCast<Geom_Curve, Curve>::eval(*chart->GetAxisCurve(x_begin, dir));

                    m_body_inter_.Init(c);

                    std::vector<Real> intersection_points;
                    for (; m_body_inter_.More(); m_body_inter_.Next()) {
                        intersection_points.push_back(m_body_inter_.W());
                    }

                    std::sort(intersection_points.begin(), intersection_points.end());

                    for (size_t n = 0; n < intersection_points.size(); n += 2) {
                        gp_Pnt p0 = c->Value(intersection_points[n]);
                        gp_Pnt p1 = c->Value(intersection_points[n + 1]);

                        point_type x0{p0.X(), p0.Y(), p0.Z()};

                        index_tuple i0{0, 0, 0};
                        point_type r0{0, 0, 0};
                        std::tie(i0, r0) = chart->invert_global_coordinates(x0);

                        point_type x1{p1.X(), p1.Y(), p1.Z()};
                        index_tuple i1{0, 0, 0};
                        point_type r1{0, 0, 0};
                        std::tie(i1, r1) = chart->invert_global_coordinates(x1);

                        index_type s0 = std::max(i0[dir], std::get<0>(m_idx_box)[dir]);
                        index_type s1 = std::min(i1[dir], std::get<1>(m_idx_box)[dir]);

                        for (index_type s = i0[dir]; s <= i1[dir]; ++s) {
                            index_tuple id{i, j, k};
                            id[dir] = s;
                            vertex_tags[0].Set(1, id);
                        }
                    }

                    // std::cout << index_tuple{i, j, k} << "~" << idx << "~" << r <<
                    // std::endl;
                    // vertex_tags->SetEntity(count, idx);
                    // std::cout << "DIR:" << dir << "\t" << m_idx_box << "\t" <<
                    // index_tuple{i, j, k} << "\t" << idx;
                    // if (!(CheckInSide(m_idx_box, idx))) {
                    //    std::cout << std::endl;
                    //    continue;
                    // } else {
                    //    std::cout << "\t" << (x) << "\t" << chart->inv_map(x) <<
                    //    std::endl;
                    // }
                    //   edge_fraction[dir].SetEntity(r[dir], idx);
                    //   vertex_tags->SetEntity(1, idx);
                    //   idx[(dir + 1) % 3] -= 1;
                    //   vertex_tags->SetEntity(1, idx);
                    //   idx[(dir + 2) % 3] -= 1;
                    //   vertex_tags->SetEntity(1, idx);
                    //   idx[(dir + 1) % 3] += 1;
                    //   vertex_tags->SetEntity(1, idx);
                    //   index_tuple id{i, j, k};
                    //   id[dir] = std::get<0>(l_coor)[dir];
                    //   vertex_tags[0].SetEntity(dir + 1, id);
                    //   id[(dir + 1) % 3] = idx[(dir + 1) % 3] - 1;
                    //   vertex_tags[0].SetEntity(dir + 1, id);
                    //   id[(dir + 2) % 3] = idx[(dir + 2) % 3] - 1;
                    //   vertex_tags[0].SetEntity(dir + 1, id);
                    //   id[(dir + 1) % 3] = idx[(dir + 1) % 3];
                    //   vertex_tags[0].SetEntity(dir + 1, id);
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
                    //   edge_fraction[dir].SetEntity(std::get<1>(l_coor)[dir], idx);
                    //

                    //                        VERBOSE << "s0:" << s0 << " s1:" << s1 << std::endl;
                    //                        if (x[dir] > std::get<1>(m_idx_box)[dir]) { break; }
                }
    }
}

}  //    namespace geometry{

}  // namespace simpla{