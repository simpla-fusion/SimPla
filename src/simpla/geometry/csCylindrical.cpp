//
// Created by salmon on 17-7-22.
//
#include "csCylindrical.h"

namespace simpla {
namespace geometry {

std::shared_ptr<GeoObject> csCylindrical::BoundBox(box_type const &b) const { return nullptr; };
std::shared_ptr<GeoObject> csCylindrical::BoundBox(index_box_type const &b) const {
    return BoundBox(std::make_tuple(local_coordinates(std::get<0>(b)), local_coordinates(std::get<0>(b))));
};

}  // namespace geometry
}  // namespace simpla
