//
// Created by salmon on 17-10-23.
//

#include "Cone.h"
#include "ShapeFunction.h"
namespace simpla {
namespace geometry {
box_type sfCone::GetParameterRange() const { return utility::make_box(m_parameter_range_); }
box_type sfCone::GetValueRange() const {
    return std::make_tuple(utility::make_point(m_value_range_[0]), utility::make_point(m_value_range_[1]));
};
SP_DEF_PARA_VALUE_RANGE(Cone)
SP_DEF_PARA_VALUE_RANGE(ConicalSurface)

SP_GEO_OBJECT_REGISTER(ConicalSurface)

void ConicalSurface::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<simpla::data::DataNode> ConicalSurface::Serialize() const {
    auto res = base_type::Serialize();
    return res;
}
ConicalSurface::ConicalSurface() = default;
ConicalSurface::ConicalSurface(ConicalSurface const &other) = default;
ConicalSurface::ConicalSurface(Axis const &axis, Real semi_angle) : ParametricSurface(axis), m_shape_(semi_angle){};

ConicalSurface::~ConicalSurface() = default;

point_type ConicalSurface::xyz(Real u, Real v) const { return m_axis_.xyz(m_shape_.Value(u, v)); };
point_type ConicalSurface::uvw(Real x, Real y, Real z) const { return m_shape_.InvValue(m_axis_.uvw(x, y, z)); };

SP_GEO_OBJECT_REGISTER(Cone)
Cone::Cone() = default;
Cone::Cone(Axis const &axis, Real semi_angle) : ParametricBody(axis), m_shape_(semi_angle){};
Cone::~Cone() = default;
std::shared_ptr<simpla::data::DataNode> Cone::Serialize() const {
    auto res = base_type::Serialize();
    //    res->SetValue("SemiAngle", m_semi_angle_);
    return res;
};
void Cone::Deserialize(std::shared_ptr<data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    //    m_semi_angle_ = cfg->GetValue("SemiAngle", m_semi_angle_);
}

point_type Cone::xyz(Real u, Real v, Real w) const { return m_axis_.xyz(m_shape_.Value(u, v, w)); };
point_type Cone::uvw(Real x, Real y, Real z) const { return m_shape_.InvValue(m_axis_.uvw(x, y, z)); };


}  // namespace geometry {
}  // namespace simpla {