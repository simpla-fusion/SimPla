//
// Created by salmon on 17-10-20.
//

#include "Plane.h"
#include "GeoAlgorithm.h"
#include "Line.h"
namespace simpla {
namespace geometry {
SP_OBJECT_REGISTER(Plane)
void Plane::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<simpla::data::DataNode> Plane::Serialize() const { return base_type::Serialize(); }

int Plane::CheckOverlap(box_type const &) const {
    UNIMPLEMENTED;
    return 0;
}

std::shared_ptr<GeoObject> Plane::Intersection(std::shared_ptr<const GeoObject> const &c, Real tolerance) const {
    if (auto line = std::dynamic_pointer_cast<const Line>(c)) {
    } else {
        UNIMPLEMENTED;
    }
    return nullptr;
};

}  // namespace geometry{
}  // namespace simpla{impla