//
// Created by salmon on 17-11-6.
//

#include "GeoEntity.h"
#include "Edge.h"
#include "Face.h"
#include "Solid.h"
namespace simpla {
namespace geometry {
GeoEntity::GeoEntity() = default;
GeoEntity::GeoEntity(GeoEntity const &) = default;
GeoEntity::~GeoEntity() = default;
std::string GeoEntity::FancyTypeName() const { return "GeoEntity"; }

}  // namespace geometry{
}  // namespace simpla{