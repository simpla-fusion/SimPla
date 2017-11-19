//
// Created by salmon on 17-11-19.
//

#include "Line.h"
namespace simpla {
namespace geometry {
SP_GEO_OBJECT_REGISTER(Line)
Line::Line(Axis const &axis) : Edge(axis, nullptr, 0, 1) {}

}  // namespace geometry {
}  // namespace simpla {