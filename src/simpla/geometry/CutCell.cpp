//
// Created by salmon on 17-10-17.
//
#ifdef OCE_FOUND
#include "occ/GeoObjectOCC.h"
#include "occ/OCECutCell.h"
#endif

#include "Box.h"
#include "Chart.h"
#include "Curve.h"
#include "CutCell.h"
#include "GeoObject.h"
namespace simpla {
namespace geometry {

#ifdef OCE_FOUND
void CutCellOCC(Array<unsigned int> *vertex_tags, std::shared_ptr<const Chart> const &chart,
                index_box_type const &m_idx_box, const std::shared_ptr<const GeoObject> &g, unsigned int tag);
#endif

void CutCellTagNodeSimPla(Array<unsigned int> *node_tags, std::shared_ptr<const Chart> const &chart,
                          index_box_type const &idx_box, const std::shared_ptr<const GeoObject> &g, unsigned int tag) {
    if (auto box = std::dynamic_pointer_cast<const Box>(g)) {
        auto bound_box = box->GetBoundingBox();
        index_tuple lo, hi;
        lo = std::get<1>(chart->invert_local_coordinates(std::get<0>(bound_box)));
        hi = std::get<1>(chart->invert_local_coordinates(std::get<1>(bound_box)));
        auto a_selection = node_tags->GetSelection(std::make_tuple(lo, hi));
        a_selection |= tag;
    }
    auto const &scale = chart->GetScale();
    Real tol = std::sqrt(dot(scale, scale) * 0.01);
    //    std::get<1>(m_idx_box) += 1;
    box_type bnd_box = g->GetBoundingBox();
    vector_type length = std::get<1>(bnd_box) - std::get<0>(bnd_box);
    std::get<0>(bnd_box) -= 0.03 * length;
    std::get<1>(bnd_box) += 0.03 * length;

    point_type xlo, xhi;
    std::tie(xlo, xhi) = bnd_box;

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
        std::tie(lo, hi) = idx_box;
        hi[dir] = lo[dir] + 1;
        for (index_type i = lo[0]; i < hi[0]; ++i)
            for (index_type j = lo[1]; j < hi[1]; ++j)
                for (index_type k = lo[2]; k < hi[2]; ++k) {
                    point_type x_begin = chart->global_coordinates(0b0, i, j, k);
                    auto curve = chart->GetAxisCurve(x_begin, dir);
                    std::vector<Real> intersection_points;
                    m_body_inter_.Init(chart->GetAxisCurve(x_begin, dir), &intersection_points);

                    for (size_t n = 0; n < intersection_points.size(); n += 2) {
                        auto p0 = curve->Value(intersection_points[n]);
                        auto p1 = curve->Value(intersection_points[n + 1]);

                        point_type x0{p0[0], p0[1], p0[2]};

                        index_tuple i0{0, 0, 0};
                        point_type r0{0, 0, 0};
                        std::tie(i0, r0) = chart->invert_global_coordinates(x0);

                        point_type x1{p1[0], p1[1], p1[2]};
                        index_tuple i1{0, 0, 0};
                        point_type r1{0, 0, 0};
                        std::tie(i1, r1) = chart->invert_global_coordinates(x1);

                        index_type s0 = std::max(i0[dir], std::get<0>(idx_box)[dir]);
                        index_type s1 = std::min(i1[dir], std::get<1>(idx_box)[dir]);

                        index_tuple id{i, j, k};
                        for (id[dir] = i0[dir]; id[dir] <= i1[dir]; ++id[dir]) {
                            node_tags->Set(node_tags->Get(id[0], id[1], id[2]) | tag, id[0], id[1], id[2]);
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
void CutCellTagNode(Array<unsigned int> *vertex_tags, std::shared_ptr<const Chart> const &chart,
                    index_box_type const &idx_box, const std::shared_ptr<const GeoObject> &g, unsigned int tag) {
    if (false) {
    }
#ifdef OCE_FOUND
    else if (auto g = std::dynamic_pointer_cast<GeoObjectOCC>(g)) {
        CutCellTagNodeOCE(vertex_tags, chart, idx_box, g, tag);
    }
#endif  // OCC_FOUND
    else {
        CutCellTagNodeSimPla(vertex_tags, chart, idx_box, g, tag);
    }
}
}  //    namespace geometry{

}  // namespace simpla{