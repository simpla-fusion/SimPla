//
// Created by salmon on 17-11-6.
//

#include "Shape.h"
namespace simpla {
namespace geometry {
Shape::Shape() = default;
Shape::Shape(Shape const &) = default;
Shape::~Shape() = default;
std::shared_ptr<Shape> Shape::Create(std::string const &key) { return Factory<Shape>::Create(key); }

}  // namespace geometry{
}  // namespace simpla{