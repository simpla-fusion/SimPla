//
// Created by salmon on 17-7-26.
//

#ifndef SIMPLA_CUTCELL_H
#define SIMPLA_CUTCELL_H

#include <simpla/algebra/Array.h>
#include <simpla/utilities/SPDefines.h>
#include <memory>
namespace simpla {
namespace geometry {

class Chart;
class GeoObject;

void CutCellTagNode(Array<unsigned int> *vertex_tags, Array<Real> *edge_tags, std::shared_ptr<const Chart> const &chart,
                    index_box_type const &idx_box, std::shared_ptr<const GeoObject> const &g, unsigned int tag = 0b001);

}  //    namespace geometry{

}  // namespace simpla{
#endif  // SIMPLA_CUTCELL_H
