//
// Created by salmon on 17-11-14.
//

#include "Face.h"
namespace simpla {
namespace geometry {
Face::Face() = default;
Face::~Face() = default;
Face::Face(Face const &other) = default;
Face::Face(Axis const &axis) : PrimitiveShape(axis) {}
void Face::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) { base_type::Deserialize(cfg); };
std::shared_ptr<simpla::data::DataNode> Face::Serialize() const { return base_type::Serialize(); };
}  // namespace geometry
}  // namespace simpla