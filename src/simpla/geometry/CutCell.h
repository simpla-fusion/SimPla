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
void CutCell(Chart *chart, index_box_type m_idx_box, GeoObject const *g, Array<Real> *vertex_tags);

}  //    namespace geometry{

}  // namespace simpla{
#endif  // SIMPLA_CUTCELL_H
