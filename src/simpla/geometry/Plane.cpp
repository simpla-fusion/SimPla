//
// Created by salmon on 17-10-20.
//

#include "Plane.h"
namespace simpla {
namespace geometry {
SP_OBJECT_REGISTER(Plane)
void Plane::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<simpla::data::DataNode> Plane::Serialize() const { return base_type::Serialize(); }

}  // namespace geometry{
}  // namespace simpla{impla