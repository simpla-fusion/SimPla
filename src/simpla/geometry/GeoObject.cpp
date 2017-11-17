//
// Created by salmon on 17-2-21.
//
#include "GeoObject.h"

#include <simpla/utilities/ParsingURI.h>
#include <memory>

#include "BoxUtilities.h"
#include "GeoAlgorithm.h"
#include "GeoEngine.h"

namespace simpla {
namespace geometry {
GeoObject::GeoObject() = default;
GeoObject::~GeoObject() = default;
GeoObject::GeoObject(GeoObject const &other) : m_axis_(other.m_axis_){};
GeoObject::GeoObject(Axis const &axis) : m_axis_(axis){};
std::shared_ptr<GeoObject> GeoObject::New(std::string const &s) {
    std::shared_ptr<GeoObject> res = nullptr;
    if (s.find(':') == std::string::npos) {
        res = Factory<GeoObject>::Create(s);
    } else {
        res = GEO_ENGINE->Load(s);
    }
    return res;
}

std::shared_ptr<GeoObject> GeoObject::New(std::shared_ptr<data::DataEntry> const &cfg) {
    auto res = Factory<GeoObject>::Create(cfg->GetValue<std::string>("_REGISTER_NAME_", ""));
    res->Deserialize(cfg);
    return res;
};
std::shared_ptr<data::DataEntry> GeoObject::Serialize() const {
    auto res = data::DataEntry::New(data::DataEntry::DN_TABLE);
    res->Set("Axis", m_axis_.Serialize());
    res->SetValue("_TYPE_", FancyTypeName());
    return res;
}
void GeoObject::Deserialize(std::shared_ptr<data::DataEntry> const &cfg) { m_axis_.Deserialize(cfg->Get("Axis")); }

int GeoObject::GetDimension() const { return 3; }
bool GeoObject::IsSimpleConnected() const { return true; }
bool GeoObject::IsConvex() const { return true; }
bool GeoObject::IsContinued() const { return true; }
bool GeoObject::IsClosed() const { return false; }
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