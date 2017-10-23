//
// Created by salmon on 17-10-21.
//
#include "Line.h"
namespace simpla {
namespace geometry {
SP_OBJECT_REGISTER(Line)

std::shared_ptr<data::DataNode> Line::Serialize() const { return base_type::Serialize(); };
void Line::Deserialize(std::shared_ptr<data::DataNode> const& cfg) { base_type::Deserialize(cfg); }

}  // namespace geometry
}  // namespace simpla