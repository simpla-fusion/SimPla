//
// Created by salmon on 17-10-21.
//
#include "spLine.h"
namespace simpla {
namespace geometry {
SP_SHAPE_REGISTER(spLine)

spLine::spLine() = default;
spLine::spLine(spLine const &) = default;
spLine::~spLine() = default;

std::shared_ptr<data::DataEntry> spLine::Serialize() const { return base_type::Serialize(); };
void spLine::Deserialize(std::shared_ptr<const data::DataEntry> const &cfg) { base_type::Deserialize(cfg); }

}  // namespace geometry
}  // namespace simpla