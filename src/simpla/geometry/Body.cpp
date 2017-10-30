//
// Created by salmon on 17-10-18.
//
#include "Body.h"
#include "Curve.h"
#include "GeoAlgorithm.h"
#include "Surface.h"

namespace simpla {
namespace geometry {
Body::Body() = default;
Body::Body(Body const &other) = default;
Body::Body(Axis const &axis) : GeoObject(axis) {}
Body::~Body() = default;
std::shared_ptr<data::DataNode> Body::Serialize() const { return base_type::Serialize(); };
void Body::Deserialize(std::shared_ptr<data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
std::tuple<bool, bool, bool> Body::IsClosed() const { return std::make_tuple(false, false, false); };
std::tuple<bool, bool, bool> Body::IsPeriodic() const { return std::make_tuple(false, false, false); };
nTuple<Real, 3> Body::GetPeriod() const { return nTuple<Real, 3>{SP_INFINITY, SP_INFINITY, SP_INFINITY}; };
nTuple<Real, 3> Body::GetMinParameter() const { return nTuple<Real, 3>{-SP_INFINITY, -SP_INFINITY, -SP_INFINITY}; }
nTuple<Real, 3> Body::GetMaxParameter() const { return nTuple<Real, 3>{SP_INFINITY, SP_INFINITY, SP_INFINITY}; }

void Body::SetParameterRange(std::tuple<nTuple<Real, 3>, nTuple<Real, 3>> const &r) {
    std::tie(m_uvw_min_, m_uvw_max_) = r;
};
void Body::SetParameterRange(nTuple<Real, 3> const &min, nTuple<Real, 3> const &max) {
    m_uvw_min_ = min;
    m_uvw_max_ = max;
};
std::tuple<nTuple<Real, 3>, nTuple<Real, 3>> Body::GetParameterRange() const {
    return std::make_tuple(m_uvw_min_, m_uvw_max_);
};
bool Body::TestInsideUVW(point_type const &uvw) const { return TestPointInsideBox(uvw, GetParameterRange()); };

box_type Body::GetBoundingBox() const {
    auto r = GetParameterRange();
    return std::make_tuple(Value(std::get<0>(r)), Value(std::get<1>(r)));
};

}  // namespace geometry
}  // namespace simpla