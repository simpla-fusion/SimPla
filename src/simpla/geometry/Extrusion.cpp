//
// Created by salmon on 17-10-24.
//

#include "Extrusion.h"
#include "Line.h"
namespace simpla {
namespace geometry {
Extrusion::Extrusion() = default;
Extrusion::Extrusion(Extrusion const &other) = default;
Extrusion::Extrusion(std::shared_ptr<const Surface> const &s, vector_type const &v) : Swept(s->GetAxis()) {}
Extrusion::~Extrusion() = default;
void Extrusion::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<simpla::data::DataNode> Extrusion::Serialize() const {
    auto res = base_type::Serialize();
    return res;
}
bool Extrusion::TestIntersection(box_type const &, Real tolerance) const {
    UNIMPLEMENTED;
    return false;
}
std::shared_ptr<GeoObject> Extrusion::Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const {
    UNIMPLEMENTED;
    return nullptr;
}

/*******************************************************************************************************************/

SP_OBJECT_REGISTER(ExtrusionSurface)
ExtrusionSurface::ExtrusionSurface() = default;
ExtrusionSurface::ExtrusionSurface(ExtrusionSurface const &other) = default;
ExtrusionSurface::ExtrusionSurface(Axis const &axis) : SweptSurface(axis) {}
ExtrusionSurface::~ExtrusionSurface() = default;

bool ExtrusionSurface::IsClosed() const { return false; };
void ExtrusionSurface::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<simpla::data::DataNode> ExtrusionSurface::Serialize() const {
    auto res = base_type::Serialize();
    return res;
}
bool ExtrusionSurface::TestIntersection(box_type const &, Real tolerance) const {
    UNIMPLEMENTED;
    return false;
}
std::shared_ptr<GeoObject> ExtrusionSurface::Intersection(std::shared_ptr<const GeoObject> const &,
                                                          Real tolerance) const {
    UNIMPLEMENTED;
    return nullptr;
}
}  // namespace geometry{
}  // namespace simpla{