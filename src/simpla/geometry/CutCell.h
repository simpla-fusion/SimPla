//
// Created by salmon on 17-7-26.
//

#ifndef SIMPLA_CUTCELL_H
#define SIMPLA_CUTCELL_H

#include <simpla/algebra/Array.h>
#include <simpla/algebra/EntityId.h>
#include <simpla/utilities/SPDefines.h>
#include <map>
namespace simpla {
namespace geometry {

class Chart;
class GeoObject;
void CutCell(Chart *chart, index_box_type const &m_idx_box, point_type const &r, GeoObject const *g,
             Range<EntityId> body_ranges[4], Range<EntityId> boundary_ranges[4], std::map<EntityId, Real> cut_cell[4],
             Array<Real> edge_fraction[3], Array<Real> *vertex_tags);

}  //    namespace geometry{

}  // namespace simpla{
#endif  // SIMPLA_CUTCELL_H
