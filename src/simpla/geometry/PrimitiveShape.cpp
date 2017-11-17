//
// Created by salmon on 17-11-6.
//

#include "PrimitiveShape.h"
#include "Body.h"
#include "Shell.h"
namespace simpla {
namespace geometry {

PrimitiveShape::PrimitiveShape() = default;
PrimitiveShape::PrimitiveShape(PrimitiveShape const &) = default;
PrimitiveShape::~PrimitiveShape() = default;
PrimitiveShape::PrimitiveShape(Axis const &axis) : Shape(axis){};

void PrimitiveShape::Deserialize(std::shared_ptr<simpla::data::DataEntry> const &cfg) { base_type::Deserialize(cfg); };
std::shared_ptr<simpla::data::DataEntry> PrimitiveShape::Serialize() const { return base_type::Serialize(); };

std::shared_ptr<Body> PrimitiveShape::AsBody() const { return Body::New(CopyThis()); };
std::shared_ptr<Shell> PrimitiveShape::AsShell() const { return Shell::New(CopyThis()); };
point_type PrimitiveShape::xyz(Real r, Real phi, Real theta) const {
    UNIMPLEMENTED;
    return point_type{SP_SNaN, SP_SNaN, SP_SNaN};
}
point_type PrimitiveShape::uvw(Real x, Real y, Real z) const {
    UNIMPLEMENTED;
    return point_type{SP_SNaN, SP_SNaN, SP_SNaN};
}
}  // namespace geometry{
}  // namespace simpla{