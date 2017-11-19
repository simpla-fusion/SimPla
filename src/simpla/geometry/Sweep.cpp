//
// Created by salmon on 17-10-24.
//

#include "Sweep.h"
#include "Circle.h"
#include "Edge.h"
#include "Line.h"
namespace simpla {
namespace geometry {

std::shared_ptr<GeoObject> MakeSweep(std::shared_ptr<const GeoObject> const &face,
                                     std::shared_ptr<const GeoObject> const &c) {
    std::shared_ptr<GeoObject> res = nullptr;
    if (auto line = std::dynamic_pointer_cast<const Line>(c)) {
    } else if (auto circle = std::dynamic_pointer_cast<const Circle>(c)) {
    } else if (auto edge = std::dynamic_pointer_cast<const Edge>(c)) {
    }
    return res;
}

}  // namespace geometry{
}  // namespace simpla{