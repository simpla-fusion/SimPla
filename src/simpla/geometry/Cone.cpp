//
// Created by salmon on 17-10-23.
//

#include "Cone.h"
#include "ShapeFunction.h"
namespace simpla {
namespace geometry {

SP_GEO_OBJECT_REGISTER(Cone)
void Cone::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<simpla::data::DataNode> Cone::Serialize() const {
    auto res = base_type::Serialize();
    return res;
}
Cone::Cone() = default;
Cone::Cone(Cone const &other) = default;
Cone::Cone(Real semi_angle) : PrimitiveShape(), m_semi_angle_(semi_angle){};
Cone::Cone(Axis const &axis, Real semi_angle) : PrimitiveShape(axis), m_semi_angle_(semi_angle){};
Cone::~Cone() = default;

}  // namespace geometry {
}  // namespace simpla {