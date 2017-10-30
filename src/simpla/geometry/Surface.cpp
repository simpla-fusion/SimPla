//
// Created by salmon on 17-10-19.
//

#include "Surface.h"
#include <simpla/SIMPLA_config.h>
#include "Curve.h"
#include "GeoAlgorithm.h"
#include "GeoObject.h"

namespace simpla {
namespace geometry {
Surface::Surface() = default;
Surface::~Surface() = default;
Surface::Surface(Surface const &other) = default;
Surface::Surface(Axis const &axis) : GeoObject(axis) {}

std::shared_ptr<data::DataNode> Surface::Serialize() const { return base_type::Serialize(); };
void Surface::Deserialize(std::shared_ptr<data::DataNode> const &cfg) { base_type::Deserialize(cfg); }

std::tuple<bool, bool> Surface::IsClosed() const { return std::make_tuple(false, false); };
std::tuple<bool, bool> Surface::IsPeriodic() const { return std::make_tuple(false, false); };
nTuple<Real, 2> Surface::GetPeriod() const { return nTuple<Real, 2>{SP_INFINITY, SP_INFINITY}; };
nTuple<Real, 2> Surface::GetMinParameter() const { return nTuple<Real, 2>{-SP_INFINITY, -SP_INFINITY}; }
nTuple<Real, 2> Surface::GetMaxParameter() const { return nTuple<Real, 2>{SP_INFINITY, SP_INFINITY}; }

void Surface::SetParameterRange(std::tuple<nTuple<Real, 2>, nTuple<Real, 2>> const &r) {
    std::tie(m_uv_min_, m_uv_max_) = r;
};
void Surface::SetParameterRange(nTuple<Real, 2> const &min, nTuple<Real, 2> const &max) {
    m_uv_min_ = min;
    m_uv_max_ = max;
};
std::tuple<nTuple<Real, 2>, nTuple<Real, 2>> Surface::GetParameterRange() const {
    return std::make_tuple(m_uv_min_, m_uv_max_);
};

point_type Surface::Value(nTuple<Real, 2> const &u) const { return Value(u[0], u[1]); };

std::shared_ptr<GeoObject> Surface::GetBoundary() const { return nullptr; }
box_type Surface::GetBoundingBox() const {
    return std::make_tuple(point_type{-SP_INFINITY, -SP_INFINITY, -SP_INFINITY},
                           point_type{-SP_INFINITY, -SP_INFINITY, -SP_INFINITY});
}
bool Surface::TestInsideUV(Real u, Real v, Real tolerance) const {
    return TestPointInsideBox(nTuple<Real, 2>{u, v}, GetParameterRange());
}

bool Surface::TestInsideUVW(point_type const &uvw, Real tolerance) const {
    return std::abs(uvw[2]) < tolerance && TestInsideUV(uvw[0], uvw[1], tolerance);
}
bool Surface::TestInside(Real x, Real y, Real z, Real tolerance) const {
    return TestInsideUVW(m_axis_.uvw(point_type{x, y, z}), tolerance);
}

bool Surface::TestInside(point_type const &x, Real tolerance) const {
    bool close_u, closed_v;
    std::tie(close_u, closed_v) = IsClosed();
    return close_u && closed_v && TestInside(x[0], x[1], x[2], tolerance);
}

bool Surface::TestIntersection(box_type const &) const {
    UNIMPLEMENTED;
    return false;
}

std::shared_ptr<GeoObject> Surface::Intersection(std::shared_ptr<const GeoObject> const &g, Real tolerance) const {
    return nullptr;
}
point_type Surface::Value(point_type const &uvw) const { return Value(uvw[0], uvw[1]); }

PointsOnSurface::PointsOnSurface() = default;
PointsOnSurface::~PointsOnSurface() = default;
PointsOnSurface::PointsOnSurface(std::shared_ptr<const Surface> const &surf) : m_surface_(surf) {}
std::shared_ptr<data::DataNode> PointsOnSurface::Serialize() const { return base_type::Serialize(); };
void PointsOnSurface::Deserialize(std::shared_ptr<data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<const Surface> PointsOnSurface::GetBasisSurface() const { return m_surface_; }
void PointsOnSurface::SetBasisSurface(std::shared_ptr<const Surface> const &c) { m_surface_ = c; }

void PointsOnSurface::PutUV(nTuple<Real, 2> uv) { m_data_.push_back(std::move(uv)); }
nTuple<Real, 2> PointsOnSurface::GetUV(size_type i) const { return m_data_[i]; }
point_type PointsOnSurface::GetPoint(size_type i) const { return m_surface_->Value(GetUV(i)); }
std::vector<nTuple<Real, 2>> const &PointsOnSurface::data() const { return m_data_; }
std::vector<nTuple<Real, 2>> &PointsOnSurface::data() { return m_data_; }
size_type PointsOnSurface::size() const { return m_data_.size(); }
point_type PointsOnSurface::Value(size_type i) const { return GetPoint(i); };
box_type PointsOnSurface::GetBoundingBox() const {
    FIXME;
    return m_surface_->GetBoundingBox();
};
}  // namespace geometry
}  // namespace simpla