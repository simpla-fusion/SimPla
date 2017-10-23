//
// Created by salmon on 17-10-18.
//

#include "Intersector.h"
#include <simpla/data/DataNode.h>
#include <simpla/utilities/SPDefines.h>
#include "Body.h"
#include "BoundedCurve.h"
#include "Box.h"
#include "GeoAlgorithm.h"
#include "Line.h"
#include "Polygon.h"
#include "Surface.h"

namespace simpla {
namespace geometry {
struct Intersector::pimpl_s {
    std::shared_ptr<const GeoObject> m_geo_;
    Real m_tolerance_ = 1.0e-6;
};
Intersector::Intersector() : m_pimpl_(new pimpl_s){};
Intersector::~Intersector() { delete m_pimpl_; };
Intersector::Intersector(std::shared_ptr<const GeoObject> const& geo, Real tolerance) : m_pimpl_(new pimpl_s) {
    m_pimpl_->m_tolerance_ = tolerance;
};
std::shared_ptr<data::DataNode> Intersector::Serialize() const { return base_type::Serialize(); }
void Intersector::Deserialize(std::shared_ptr<data::DataNode> const& cfg) { base_type::Deserialize(cfg); };
void Intersector::SetGeoObject(std::shared_ptr<const GeoObject> const& geo) { m_pimpl_->m_geo_ = geo; }
std::shared_ptr<const GeoObject> Intersector::SetGeoObject() const { return m_pimpl_->m_geo_; }
void Intersector::SetTolerance(Real tolerance) { m_pimpl_->m_tolerance_ = tolerance; }
Real Intersector::GetTolerance() const { return m_pimpl_->m_tolerance_; }
std::shared_ptr<Intersector> Intersector::New(std::shared_ptr<const GeoObject> const& geo, Real tolerance) {
    std::shared_ptr<Intersector> res = nullptr;
    if (geo->ClassName() == "GeoObjectOCE") {
        res = std::dynamic_pointer_cast<Intersector>(Create("IntersectorOCE"));
        if (res != nullptr) {
            res->SetGeoObject(geo);
            res->SetTolerance(tolerance);
        }
    } else {
        res = std::shared_ptr<Intersector>(new Intersector(geo, tolerance));
    }

    return res;
}
size_type Intersector::GetIntersectionPoints(std::shared_ptr<const GeoObject> const& curve,
                                             std::vector<Real>& intersection_point) const {
    size_type count = 0;
    if (auto line = std::dynamic_pointer_cast<const BoundedCurve<Line>>(curve)) {
        point_type const& l0 = line->GetStartPoint();
        point_type const& l1 = line->GetEndPoint();

        //        for (auto const& obj : m_pimpl_->m_objs_) {
        //            if (!isOverlapped(obj->GetBoundingBox(), std::make_tuple(l0, l1))) { continue; }
        //            if (auto plane = std::dynamic_pointer_cast<Plane>(obj)) {
        //                Real dist, s, u, v;
        //                std::tie(dist, s, u, v) = NearestPointLineToPlane(l0, l1, plane->GetVertices()[0],
        //                                                                  plane->GetVertices()[1],
        //                                                                  plane->GetVertices()[2]);
        //                if (std::abs(dist) > m_pimpl_->m_tolerance_ || s < 0 || s > 1 ||
        //                    !plane->CheckInsideUV(u, v, m_pimpl_->m_tolerance_)) {
        //                    continue;
        //                } else {
        //                    intersection_point.push_back(s);
        //                }
        //            } else {
        //                UNIMPLEMENTED;
        //            }
        //        }
    }
    return count;
}
// size_type Intersector::GetIntersectionCurve(std::shared_ptr<const Surface> const& curve,
//                                            std::vector<std::shared_ptr<Curve>>) {}
// size_type Intersector::GetIntersectionBody(std::shared_ptr<const Body> const& curve,
//                                           std::vector<std::shared_ptr<Body>>) {}

}  // namespace geometry {
}  // namespace simpla