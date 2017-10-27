//
// Created by salmon on 17-10-26.
//

#include "PointList.h"
#include "Line.h"

namespace simpla {
namespace geometry {
SP_OBJECT_REGISTER(PointList)
void PointList::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<simpla::data::DataNode> PointList::Serialize() const { return base_type::Serialize(); }

}  // namespace geometry{
}  // namespace simpla{impla