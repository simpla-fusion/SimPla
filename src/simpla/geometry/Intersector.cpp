//
// Created by salmon on 17-10-18.
//

#include "Intersector.h"
#include <simpla/data/DataNode.h>
#include <simpla/utilities/SPDefines.h>
#include "Body.h"
#include "Box.h"
#include "Surface.h"

namespace simpla {
namespace geometry {
struct Intersector::pimpl_s {
    std::shared_ptr<const GeoObject> m_surface_;
    Real m_tolerance_ = 1.0e-6;
};
Intersector::Intersector() : m_pimpl_(new pimpl_s){};
Intersector::~Intersector() { delete m_pimpl_; };
Intersector::Intersector(std::shared_ptr<const GeoObject> const& geo, Real tolerance) : m_pimpl_(new pimpl_s) {
    m_pimpl_->m_surface_ = geo;
    m_pimpl_->m_tolerance_ = tolerance;
};
std::shared_ptr<data::DataNode> Intersector::Serialize() const { return base_type::Serialize(); }
void Intersector::Deserialize(std::shared_ptr<data::DataNode> const& cfg) { base_type::Deserialize(cfg); };
void Intersector::SetGeoObject(std::shared_ptr<const GeoObject> const& geo) { m_pimpl_->m_surface_ = geo; }
std::shared_ptr<const GeoObject> Intersector::SetGeoObject() const { return m_pimpl_->m_surface_; }
void Intersector::SetTolerance(Real tolerance) { m_pimpl_->m_tolerance_ = tolerance; }
Real Intersector::GetTolerance() const { return m_pimpl_->m_tolerance_; }
std::shared_ptr<Intersector> Intersector::New(std::shared_ptr<const GeoObject> const& geo, Real tolerance) {
    std::shared_ptr<Intersector> res;
    if (geo->ClassName() == "GeoObjectOCE") {
        res = std::dynamic_pointer_cast<Intersector>(Create("IntersectorOCE"));
        res->SetGeoObject(geo);
        res->SetTolerance(tolerance);
    } else if (auto g = std::dynamic_pointer_cast<const Body>(geo)) {
        res = std::shared_ptr<Intersector>(new Intersector(g->GetBoundary(), tolerance));
    } else if (auto g = std::dynamic_pointer_cast<const Surface>(geo)) {
        res = std::shared_ptr<Intersector>(new Intersector(g, tolerance));
    } else {
        UNIMPLEMENTED;
    }

    return res;
}
size_type Intersector::GetIntersectionPoints(std::shared_ptr<const GeoObject> const& curve,
                                             std::vector<Real>& intersection_point) const {
    size_type count = 0;

    return count;
}
// size_type Intersector::GetIntersectionCurve(std::shared_ptr<const Surface> const& curve,
//                                            std::vector<std::shared_ptr<Curve>>) {}
// size_type Intersector::GetIntersectionBody(std::shared_ptr<const Body> const& curve,
//                                           std::vector<std::shared_ptr<Body>>) {}

}  // namespace geometry {
}  // namespace simpla