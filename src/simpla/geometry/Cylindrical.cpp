//
// Created by salmon on 17-7-22.
//

#include "Cylindrical.h"
#include "PointsOnCurve.h"
namespace simpla {
namespace geometry {
SP_DEF_SHAPE_FUNCTION_PARA_VALUE_RANGE(Cylindrical)
SP_DEF_PARA_VALUE_RANGE(Cylindrical)
SP_DEF_PARA_VALUE_RANGE(CylindricalSurface)

std::shared_ptr<simpla::data::DataNode> Cylindrical::Serialize() const { return base_type::Serialize(); };
void Cylindrical::Deserialize(std::shared_ptr<data::DataNode> const &cfg) { base_type::Deserialize(cfg); }

point_type Cylindrical::xyz(Real u, Real v, Real w) const { return m_axis_.xyz(m_shape_.Value(u, v, w)); };
point_type Cylindrical::uvw(Real x, Real y, Real z) const { return m_shape_.InvValue(m_axis_.uvw(x, y, z)); };

bool Cylindrical::CheckIntersection(point_type const &p, Real tolerance) const {
    return m_shape_.Distance(m_axis_.uvw(p)) < 0;
};
bool Cylindrical::CheckIntersection(box_type const &b, Real tolerance) const {
    return m_shape_.TestBoxGetIntersectionion(m_axis_.uvw(std::get<0>(b)), m_axis_.uvw(std::get<1>(b)));
};
std::shared_ptr<Body> Cylindrical::GetIntersection(std::shared_ptr<const Body> const &, Real tolerance) const {
    UNIMPLEMENTED;
    return nullptr;
};
std::shared_ptr<Curve> Cylindrical::GetIntersection(std::shared_ptr<const Curve> const &, Real tolerance) const {
    UNIMPLEMENTED;
    return nullptr;
};
std::shared_ptr<Surface> Cylindrical::GetIntersection(std::shared_ptr<const Surface> const &, Real tolerance) const {
    UNIMPLEMENTED;
    return nullptr;
};
/**********************************************************************************************************************/

SP_OBJECT_REGISTER(CylindricalSurface)
CylindricalSurface::CylindricalSurface() = default;
CylindricalSurface::CylindricalSurface(CylindricalSurface const &other) = default;
CylindricalSurface::CylindricalSurface(Axis const &axis, Real radius) : ParametricSurface(axis), m_radius_(radius) {}

CylindricalSurface::~CylindricalSurface() = default;
void CylindricalSurface::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    m_radius_ = cfg->GetValue<Real>("Radius", m_radius_);
}
std::shared_ptr<simpla::data::DataNode> CylindricalSurface::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue<Real>("Radius", m_radius_);
    return res;
}

point_type CylindricalSurface::xyz(Real u, Real v) const { return m_axis_.xyz(m_shape_.Value(u, v)); };
point_type CylindricalSurface::uvw(Real x, Real y, Real z) const { return m_shape_.InvValue(m_axis_.uvw(x, y, z)); };
bool CylindricalSurface::CheckIntersection(box_type const &b, Real tolerance) const {
    return m_shape_.TestBoxGetIntersectionion(m_axis_.uvw(std::get<0>(b)), m_axis_.uvw(std::get<1>(b)));
}

std::shared_ptr<PointsOnCurve> CylindricalSurface::GetIntersection(std::shared_ptr<const Curve> const &g,
                                                                   Real tolerance) const {
    return base_type::GetIntersection(g, tolerance);
};
std::shared_ptr<Curve> CylindricalSurface::GetIntersection(std::shared_ptr<const Surface> const &g,
                                                           Real tolerance) const {
    return base_type::GetIntersection(g, tolerance);
}
}  // namespace geometry {
}  // namespace simpla {
