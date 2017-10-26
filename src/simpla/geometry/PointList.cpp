//
// Created by salmon on 17-10-26.
//

#include "PointList.h"
#include "Line.h"

namespace simpla {
namespace geometry {
SP_OBJECT_REGISTER(PointList)
void PointList::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<simpla::data::DataNode> PointList::Serialize() const { return base_type::Serialize(); }

int PointList::CheckOverlap(box_type const &) const {
    UNIMPLEMENTED;
    return 0;
}

std::shared_ptr<GeoObject> PointList::Intersection(std::shared_ptr<const GeoObject> const &c,
                                                       Real tolerance) const const {
    if (auto line = std::dynamic_pointer_cast<const Line>(c)) {
    } else {
        UNIMPLEMENTED;
    }
    return nullptr;
};

}  // namespace geometry{
}  // namespace simpla{impla