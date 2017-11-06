//
// Created by salmon on 17-10-24.
//

#include "Swept.h"
namespace simpla {
namespace geometry {

Swept::Swept() = default;
Swept::Swept(Swept const &other) = default;
Swept::Swept(Axis const &axis) : PrimitiveShape(axis) {}
Swept::~Swept() = default;
void Swept::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<simpla::data::DataNode> Swept::Serialize() const { return base_type::Serialize(); }

}  // namespace geometry{
}  // namespace simpla{