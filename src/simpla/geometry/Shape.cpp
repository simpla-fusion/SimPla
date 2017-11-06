//
// Created by salmon on 17-11-6.
//

#include "Shape.h"
namespace simpla {
namespace geometry {
Shape::Shape() = default;
Shape::Shape(Shape const &) = default;
Shape::~Shape() = default;
Shape::Shape(Axis const &axis) : GeoObject(axis){};
void Shape::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) { base_type::Deserialize(cfg); };
std::shared_ptr<simpla::data::DataNode> Shape::Serialize() const {
    auto res = base_type::Serialize();
    return res;
};
}  // namespace geometry{
}  // namespace simpla{