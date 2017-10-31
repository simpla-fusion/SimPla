//
// Created by salmon on 17-7-26.
//

#ifndef SIMPLA_CURVE_H
#define SIMPLA_CURVE_H

#include <simpla/utilities/Constants.h>
#include "Axis.h"
#include "GeoObject.h"
#include "Plane.h"

namespace simpla {
namespace geometry {
struct Box;
struct Curve : public GeoObject {
    SP_GEO_ABS_OBJECT_HEAD(Curve, GeoObject);

   protected:
    Curve();
    Curve(Curve const &other);
    explicit Curve(Axis const &axis);

   public:
    ~Curve() override;

    virtual bool IsClosed() const;
    box_type GetBoundingBox() const override;
    virtual std::shared_ptr<PolyPoints> GetBoundaryPoints() const;
    virtual std::shared_ptr<PolyPoints> Intersection(std::shared_ptr<const Curve> const &g, Real tolerance) const;
    virtual std::shared_ptr<PolyPoints> Intersection(std::shared_ptr<const Plane> const &g, Real tolerance) const;
    virtual std::shared_ptr<GeoObject> Intersection(std::shared_ptr<const Box> const &g, Real tolerance) const;
    std::shared_ptr<GeoObject> Intersection(std::shared_ptr<const GeoObject> const &g, Real tolerance) const override;

    bool TestIntersection(point_type const &, Real tolerance) const override;
    bool TestIntersection(box_type const &, Real tolerance) const override;

    std::shared_ptr<GeoObject> GetBoundary() const override;
};

// struct PolyPoints;
// struct PointsOnCurve : public PolyPoints {
//    SP_GEO_OBJECT_HEAD(PointsOnCurve, PolyPoints);
//
//   protected:
//    PointsOnCurve();
//    explicit PointsOnCurve(std::shared_ptr<const Curve> const &);
//
//   public:
//    ~PointsOnCurve() override;
//    Axis const &GetAxis() const override;
//    std::shared_ptr<const Curve> GetBasisCurve() const;
//    void SetBasisCurve(std::shared_ptr<const Curve> const &c);
//
//    void PutU(Real);
//    Real GetU(size_type i) const;
//    point_type GetPoint(size_type i) const;
//    std::vector<Real> const &data() const;
//    std::vector<Real> &data();
//
//    size_type size() const override;
//    point_type Value(size_type i) const override;
//    box_type GetBoundingBox() const override;
//
//   private:
//    std::shared_ptr<const Curve> m_curve_;
//    std::vector<Real> m_data_;
//};
}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_CURVE_H
