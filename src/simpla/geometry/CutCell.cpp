//
// Created by salmon on 17-10-17.
//
#ifdef OCE_FOUND
#include "occ/GeoObjectOCC.h"
#include "occ/OCECutCell.h"
#endif

#include "Box.h"
#include "Chart.h"
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