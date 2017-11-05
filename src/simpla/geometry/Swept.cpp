//
// Created by salmon on 17-10-24.
//

#include "Swept.h"
namespace simpla {
namespace geometry {
SP_GEO_OBJECT_REGISTER(Swept)

Swept::Swept() = default;
Swept::Swept(Swept const &other) = default;
Swept::Swept(Axis const &axis) : Body(axis) {}
Swept::~Swept() = default;
void Swept::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<simpla::data::DataNode> Swept::Serialize() const { return base_type::Serialize(); }

/*******************************************************************************************************************/

SweptSurface::SweptSurface() = default;
SweptSurface::SweptSurface(SweptSurface const &other) = default;
SweptSurface::SweptSurface(Axis const &axis) : Surface(axis) {}
SweptSurface::~SweptSurface() = default;
void SweptSurface::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<simpla::data::DataNode> SweptSurface::Serialize() const { return base_type::Serialize(); }

}  // namespace geometry{
}  // namespace simpla{