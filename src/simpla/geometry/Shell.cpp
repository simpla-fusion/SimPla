//
// Created by salmon on 17-11-6.
//

#include "Shell.h"
#include "GeoEntity.h"
namespace simpla {
namespace geometry {
Shell::Shell(Shell const &) = default;
Shell::Shell(Axis const &axis) : GeoObject(axis){};
void Shell::Deserialize(std::shared_ptr<const simpla::data::DataEntry> const &cfg) { base_type::Deserialize(cfg); };
std::shared_ptr<simpla::data::DataEntry> Shell::Serialize() const { return base_type::Serialize(); };
}  // namespace geometry{
}  // namespace simpla{