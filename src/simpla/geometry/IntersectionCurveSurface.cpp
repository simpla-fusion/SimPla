//
// Created by salmon on 17-11-2.
//

#include "IntersectionCurveSurface.h"
#include <simpla/utilities/Factory.h>
#include <simpla/utilities/Log.h>
#include <vector>
#include "GeoEngine.h"
#include "PointsOnCurve.h"
#include "Shell.h"
#include "Surface.h"
namespace simpla {
namespace geometry {

IntersectionCurveSurface::IntersectionCurveSurface() = default;
IntersectionCurveSurface::~IntersectionCurveSurface() = default;
std::shared_ptr<IntersectionCurveSurface> IntersectionCurveSurface::New(std::string const &key) {
    return Factory<IntersectionCurveSurface>::Create(key.empty() ? GeoEngine::RegisterName() : key);
}

void IntersectionCurveSurface::SetUp(std::shared_ptr<const Surface> const &g, Real tolerance) {
    m_surface_ = g;
    m_tolerance_ = tolerance;
}

void IntersectionCurveSurface::TearDown() { m_surface_.reset(); }
size_type IntersectionCurveSurface::Intersect(std::shared_ptr<const Curve> const &curve, std::vector<Real> *p) const {
    ASSERT(p != nullptr);
    size_type count = 0;
    if (auto points = std::dynamic_pointer_cast<PointsOnCurve>(m_surface_->GetIntersection(curve, m_tolerance_))) {
        for (auto const &v : points->data()) { p->push_back(v); }
        count = points->data().size();
    }
    return count;
}

}  //    namespace geometry{
}  // namespace simpla{