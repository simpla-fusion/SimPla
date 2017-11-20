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
std::string GeoEntity::FancyTypeName() const override { return "GeoEntity"; }
std::shared_ptr<GeoEntity> GeoEntity::Create(std::string const &k) { return simpla::Factory<GeoEntity>::Create(k); }
std::shared_ptr<GeoEntity> GeoEntity::Create(std::shared_ptr<const simpla::data::DataEntry> const &cfg) {
    auto res = simpla::Factory<GeoEntity>::Create(cfg->GetValue<std::string>("_REGISTER_NAME_", ""));
    res->Deserialize(cfg);
    return res;
}
}  // namespace geometry{
}  // namespace simpla{