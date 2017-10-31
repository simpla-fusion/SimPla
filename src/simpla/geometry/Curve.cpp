//
// Created by salmon on 17-10-18.
//
#include "Curve.h"
#include "Body.h"
#include "Box.h"
#include "PolyPoints.h"
#include "Surface.h"
namespace simpla {
namespace geometry {
Curve::Curve() = default;
Curve::~Curve() = default;
Curve::Curve(Curve const &other) = default;
Curve::Curve(Axis const &axis) : GeoObject(axis) {}

std::shared_ptr<simpla::data::DataNode> Curve::Serialize() const { return base_type::Serialize(); };
void Curve::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) { base_type::Deserialize(cfg); };
bool Curve::IsClosed() const { return false; }
std::shared_ptr<GeoObject> Curve::GetBoundary() const {
    return std::dynamic_pointer_cast<GeoObject>(GetBoundaryPoints());
}
box_type Curve::GetBoundingBox() const override {
    UNIMPLEMENTED;
    return nullptr;
}
std::shared_ptr<PolyPoints> Curve::GetBoundaryPoints() const {
    UNIMPLEMENTED;
    return nullptr;
}
std::shared_ptr<PolyPoints> Curve::Intersection(std::shared_ptr<const Curve> const &g, Real tolerance) const {
    UNIMPLEMENTED;
    return nullptr;
}
std::shared_ptr<PolyPoints> Curve::Intersection(std::shared_ptr<const Plane> const &g, Real tolerance) const {
    UNIMPLEMENTED;
    return nullptr;
}

std::shared_ptr<GeoObject> Curve::Intersection(std::shared_ptr<const Box> const &g, Real tolerance) const {
    UNIMPLEMENTED;
    return nullptr;
}
bool Curve::TestIntersection(point_type const &, Real tolerance) const override {
    UNIMPLEMENTED;
    return false;
}
bool Curve::TestIntersection(box_type const &, Real tolerance) const override {
    UNIMPLEMENTED;
    return false;
}

std::shared_ptr<GeoObject> Curve::Intersection(std::shared_ptr<const GeoObject> const &g, Real tolerance) const {
    std::shared_ptr<GeoObject> res = nullptr;
    if (auto curve = std::dynamic_pointer_cast<const Curve>(g)) {
        res = Intersection(curve, tolerance);
    } else if (auto plane = std::dynamic_pointer_cast<const Plane>(g)) {
        res = Intersection(plane, tolerance);
    } else if (auto box = std::dynamic_pointer_cast<const Box>(g)) {
        res = Intersection(box, tolerance);
    } else if (auto surface = std::dynamic_pointer_cast<const Surface>(g)) {
        res = surface->Intersection(std::dynamic_pointer_cast<const Curve>(shared_from_this()), tolerance);
    } else if (auto body = std::dynamic_pointer_cast<const Body>(g)) {
        res = body->Intersection(std::dynamic_pointer_cast<const Curve>(shared_from_this()), tolerance);
    } else {
        UNIMPLEMENTED;
    }
    return res;
};
///*box_type Curve::GetBoundingBox() const {
//    return std::make_tuple(point_type{-SP_INFINITY, -SP_INFINITY, -SP_INFINITY},
//                           point_type{-SP_INFINITY, -SP_INFINITY, -SP_INFINITY});
//}*/
// bool Curve::TestIntersection(box_type const &) const { return false; }
// bool Curve::TestInside(point_type const &x, Real tolerance) const { return false; }
// bool Curve::TestInsideU(Real u) const { return m_u_min_ <= u && u <= m_u_max_; }
// bool Curve::TestInsideUVW(point_type const &uvw, Real tolerance) const { return TestInsideU(uvw[0]); }
// std::shared_ptr<GeoObject> Curve::Intersection(std::shared_ptr<const GeoObject> const &g, Real tolerance) const {
//    UNIMPLEMENTED;
//    return nullptr;
//}
// point_type Curve::Value(point_type const &uvw) const { return Value(uvw[0]); }
//
// PointsOnCurve::PointsOnCurve() = default;
// PointsOnCurve::~PointsOnCurve() = default;
// PointsOnCurve::PointsOnCurve(std::shared_ptr<const Curve> const &curve)
//    : PolyPoints(curve->GetAxis()), m_curve_(std::dynamic_pointer_cast<Curve>(curve->Copy())) {}
// std::shared_ptr<data::DataNode> PointsOnCurve::Serialize() const { return base_type::Serialize(); };
// void PointsOnCurve::Deserialize(std::shared_ptr<data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
// Axis const &PointsOnCurve::GetAxis() const { return m_curve_->GetAxis(); };
// std::shared_ptr<const Curve> PointsOnCurve::GetBasisCurve() const { return m_curve_; }
// void PointsOnCurve::SetBasisCurve(std::shared_ptr<const Curve> const &c) { m_curve_ = c; }
//
// void PointsOnCurve::PutU(Real u) { m_data_.push_back(u); }
// Real PointsOnCurve::GetU(size_type i) const { return m_data_[i]; }
// point_type PointsOnCurve::GetPoint(size_type i) const { return m_curve_->Value(GetU(i)); }
// size_type PointsOnCurve::size() const { return m_data_.size(); }
// std::vector<Real> const &PointsOnCurve::data() const { return m_data_; }
// std::vector<Real> &PointsOnCurve::data() { return m_data_; }
//
// point_type PointsOnCurve::Value(size_type i) const { return GetPoint(i); };
// box_type PointsOnCurve::GetBoundingBox() const {
//    FIXME;
//    return m_curve_->GetBoundingBox();
//};
////
////Circle::Circle() = default;
////
////std::shared_ptr<simpla::data::DataNode> Circle::Serialize() const {
////    auto cfg = base_type::Serialize();
////    cfg->SetValue("Origin", m_origin_);
////    cfg->SetValue("Radius", m_radius_);
////    cfg->SetValue("Normal", m_normal_);
////    cfg->SetValue("R", m_r_);
////    cfg->SetValue("Radius", m_radius_);
////    return cfg;
////};
////void Circle::Deserialize(std::shared_ptr<simpla::data::DataNode> const& cfg) { base_type::Deserialize(cfg); };
////Arc::Arc() = default;
////Arc::~Arc() = default;
////std::shared_ptr<simpla::data::DataNode> Arc::Serialize() const {
////    auto cfg = base_type::Serialize();
////    cfg->SetValue("Origin", m_origin_);
////    cfg->SetValue("Radius", m_radius_);
////    cfg->SetValue("AngleBegin", m_angle_begin_);
////    cfg->SetValue("AngleEnd", m_angle_end_);
////    cfg->SetValue("XAxis", m_XAxis_);
////    cfg->SetValue("YAXis", m_YAxis_);
////    return cfg;
////};
////void Arc::Deserialize(std::shared_ptr<simpla::data::DataNode> const& cfg) {
////    base_type::Deserialize(cfg);
////    m_origin_ = cfg->GetValue("Origin", m_origin_);
////    m_radius_ = cfg->GetValue("Radius", m_radius_);
////    m_angle_begin_ = cfg->GetValue("AngleBegin", m_angle_begin_);
////    m_angle_end_ = cfg->GetValue("AngleEnd", m_angle_end_);
////    m_XAxis_ = cfg->GetValue("XAxis", m_XAxis_);
////    m_YAxis_ = cfg->GetValue("YAXis", m_YAxis_);
////};
// Line::Line() = default;
// Line::~Line() = default;
// std::shared_ptr<simpla::data::DataNode> Line::Serialize() const {
//    auto cfg = base_type::Serialize();
//    cfg->SetValue("Begin", m_p0_);
//    cfg->SetValue("End", m_p1_);
//    return cfg;
//};
// void Line::Deserialize(std::shared_ptr<simpla::data::DataNode> const& cfg) {
//    base_type::Deserialize(cfg);
//    m_p0_ = cfg->GetValue("Begin", m_p0_);
//    m_p1_ = cfg->GetValue("End", m_p1_);
//};
//
// AxeLine::AxeLine() = default;
// AxeLine::~AxeLine() = default;
// std::shared_ptr<simpla::data::DataNode> AxeLine::Serialize() const {
//    auto cfg = base_type::Serialize();
//    cfg->SetValue("Direction", m_dir_);
//    return cfg;
//};
// void AxeLine::Deserialize(std::shared_ptr<simpla::data::DataNode> const& cfg) {
//    base_type::Deserialize(cfg);
//    m_dir_ = cfg->GetValue("Direction", m_dir_);
//};
}  // namespace geometry
}  // namespace simpla