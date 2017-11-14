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
}  // namespace geometry
}  // namespace simpla