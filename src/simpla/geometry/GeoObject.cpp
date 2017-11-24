//
// Created by salmon on 17-2-21.
//
#include "GeoObject.h"

#include <simpla/utilities/ParsingURI.h>
#include <memory>
#include "BoxUtilities.h"
#include "GeoAlgorithm.h"
#include "GeoEngine.h"
#include "GeoEntity.h"
namespace simpla {
namespace geometry {
GeoObject::GeoObject() = default;
GeoObject::~GeoObject() = default;
GeoObject::GeoObject(GeoObject const &other) : data::Configurable(other), m_axis_(other.m_axis_) {}
GeoObject::GeoObject(Axis const &axis) : m_axis_(axis){};
std::shared_ptr<const GeoObject> GeoObject::Self() const { return shared_from_this(); }
std::shared_ptr<GeoObject> GeoObject::Self() { return shared_from_this(); }

std::shared_ptr<data::DataEntry> GeoObject::Serialize() const {
    auto res = base_type::Serialize();
    res->Set("Axis", m_axis_.Serialize());
    return res;
}
void GeoObject::Deserialize(std::shared_ptr<const data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
    m_axis_.Deserialize(cfg->Get("Axis"));
}

Axis &GeoObject::GetAxis() { return m_axis_; }
Axis const &GeoObject::GetAxis() const { return m_axis_; }
void GeoObject::SetAxis(Axis const &a) { m_axis_ = a; }

void GeoObject::Mirror(const point_type &p) { m_axis_.Mirror(p); }
void GeoObject::Mirror(const Axis &a1) { m_axis_.Mirror(a1); }
void GeoObject::Rotate(const Axis &a1, Real angle) { m_axis_.Rotate(a1, angle); }
void GeoObject::Scale(Real s, int dir) { m_axis_.Scale(s, dir); }
void GeoObject::Translate(const vector_type &v) { m_axis_.Translate(v); }
void GeoObject::Move(const point_type &p) { m_axis_.Move(p); }

void GeoObject::Transform(const Axis &axis) { SetAxis(axis); }
std::shared_ptr<GeoObject> GeoObject::Transformed(const Axis &axis) const {
    auto res = Copy();
    res->Transform(axis);
    return res;
}

std::shared_ptr<GeoObject> GeoObject::GetBoundary() const { return GEO_ENGINE->GetBoundary(Self()); }
box_type GeoObject::GetBoundingBox() const {
    return box_type{{-SP_INFINITY, -SP_INFINITY, -SP_INFINITY}, {SP_INFINITY, SP_INFINITY, SP_INFINITY}};
}
bool GeoObject::CheckIntersection(point_type const &p, Real tolerance) const {
    return GEO_ENGINE->CheckIntersection(Self(), p, tolerance);
}
bool GeoObject::CheckIntersection(box_type const &b, Real tolerance) const {
    return GEO_ENGINE->CheckIntersection(Self(), b, tolerance);
}
bool GeoObject::CheckIntersection(std::shared_ptr<const GeoObject> const &g, Real tolerance) const {
    return GEO_ENGINE->CheckIntersection(Self(), g, tolerance);
}

std::shared_ptr<GeoObject> GeoObject::GetUnion(std::shared_ptr<const GeoObject> const &g, Real tolerance) const {
    return GEO_ENGINE->GetUnion(Self(), g, tolerance);
}
std::shared_ptr<GeoObject> GeoObject::GetDifference(std::shared_ptr<const GeoObject> const &g, Real tolerance) const {
    return GEO_ENGINE->GetDifference(Self(), g, tolerance);
}
std::shared_ptr<GeoObject> GeoObject::GetIntersection(std::shared_ptr<const GeoObject> const &g, Real tolerance) const {
    return GEO_ENGINE->GetIntersection(Self(), g, tolerance);
}

GeoObjectHandle::GeoObjectHandle(std::shared_ptr<const GeoEntity> const &geo, Axis const &axis, box_type const &range)
    : GeoObject(axis), m_geo_entity_(geo), m_ParameterRange_(range) {}
void GeoObjectHandle::Deserialize(std::shared_ptr<const simpla::data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
    m_geo_entity_ = GeoEntity::Create(cfg->Get("Geometry"));
}
std::shared_ptr<simpla::data::DataEntry> GeoObjectHandle::Serialize() const {
    auto res = base_type::Serialize();
    res->Set("Geometry", m_geo_entity_->Serialize());
    return res;
};
std::shared_ptr<const GeoEntity> GeoObjectHandle::GetBasisGeometry() const { return m_geo_entity_; }
void GeoObjectHandle::SetBasisGeometry(std::shared_ptr<const GeoEntity> const &g) { m_geo_entity_ = g; }
// std::shared_ptr<GeoObject> GeoObject::GetBoundary() const { return nullptr; }

// Real GeoObject::Measure() const {
//    auto b = GetBoundingBox();
//    return (std::get<1>(b)[0] - std::get<0>(b)[0]) * (std::get<1>(b)[1] - std::get<0>(b)[1]) *
//           (std::get<1>(b)[2] - std::get<0>(b)[2]);
//};
//
// bool GeoObject::IsInside(point_type const &x, Real tolerance) const {
//    return geometry::isInSide(GetBoundingBox(), x);
//}
// std::shared_ptr<GeoObject> GeoObject::GetIntersectionion(std::shared_ptr<GeoObject> const &other) const {
//    return Cube::New(geometry::Overlap(GetBoundingBox(), other->GetBoundingBox()));
//}
// std::shared_ptr<GeoObject> GeoObject::Difference(std::shared_ptr<GeoObject> const &other) const {
//    UNIMPLEMENTED;
//    return nullptr;
//}
// std::shared_ptr<GeoObject> GeoObject::Union(std::shared_ptr<GeoObject> const &other) const {
//    return Cube::New(geometry::Union(GetBoundingBox(), other->GetBoundingBox()));
//}
// Real GeoObject::isOverlapped(box_type const &b) const { return Measure(Overlap(GetBoundingBox(), b)) / measure(); }
//
// Real GeoObject::CheckIntersection(GeoObject const &other) const { return isOverlapped(other.GetBoundingBox()); }
//
// bool GeoObject::IsInside(const point_type &x) const { return CheckInSide(GetBoundingBox(), x); };
//
// std::tuple<Real, point_type, point_type> GeoObject::ClosestPoint(point_type const &x) const {
//    return std::tuple<Real, point_type, point_type>{0, x, x};
//}

}  // namespace geometry {
}  // namespace simpla {