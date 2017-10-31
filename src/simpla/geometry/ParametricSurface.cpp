//
// Created by salmon on 17-10-31.
//

#include "ParametricSurface.h"
#include "GeoAlgorithm.h"
#include "Line.h"
#include "ShapeFunction.h"
namespace simpla {
namespace geometry {
ParametricSurface::ParametricSurface() = default;
ParametricSurface::ParametricSurface(ParametricSurface const &other) = default;
ParametricSurface::ParametricSurface(Axis const &axis) : GeoObject(axis) {}
ParametricSurface::~ParametricSurface() = default;
bool ParametricSurface::IsClosed() const {return false};

std::shared_ptr<data::DataNode> ParametricSurface::Serialize() const { return base_type::Serialize(); };
void ParametricSurface::Deserialize(std::shared_ptr<data::DataNode> const &cfg) { base_type::Deserialize(cfg); }

std::shared_ptr<Curve> ParametricSurface::GetBoundaryCurve() const {
    return std::dynamic_pointer_cast<Curve>(m_curve_);
}
box_type ParametricSurface::GetBoundingBox() const { return GetValueRange(); };

point_type ParametricSurface::xyz(point_type const &u) const { return xyz(u[0], u[1]); }
point_type ParametricSurface::uvw(point_type const &x) const { return uvw(x[0], x[1], x[2]); };
std::shared_ptr<PolyPoints> ParametricSurface::Intersection(std::shared_ptr<const Curve> const &g,
                                                            Real tolerance) const {
    std::shared_ptr<PolyPoints> res = nullptr;
    if (auto line = std::dynamic_pointer_cast<const Line>(g)) {
        //        auto p_on_line = PointsOnline::New(line);
        //        auto num = shape().LineIntersection(p_on_line->StartPoint().o, p_on_line->EndPoint(), nullptr);
        //        p_on_line->resize(num);
        //        shape().LineIntersection(p_on_line->StartPoint().o, p_on_line->EndPoint(), &p_on_line->data()[0]);
        //        res = std::dynamic_pointer_cast<PolyPoints>(p_on_line);
    }
    return nullptr;
}

std::shared_ptr<Curve> ParametricSurface::Intersection(std::shared_ptr<const Surface> const &g, Real tolerance) const {
    return nullptr;
}

bool ParametricSurface::TestIntersection(point_type const &p, Real tolerance) const {
    return shape().Distance(m_axis_.uvw(p)) < 0;
};
bool ParametricSurface::TestIntersection(box_type const &box, Real tolerance) const {
    return shape().TestBoxIntersection(m_axis_.uvw(std::get<0>(box)), m_axis_.uvw(std::get<1>(box)));
};

}  // namespace geometry
}  // namespace simpla