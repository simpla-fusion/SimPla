//
// Created by salmon on 17-10-23.
//

#ifndef SIMPLA_SPHERE_H
#define SIMPLA_SPHERE_H

#include <simpla/SIMPLA_config.h>
#include <simpla/utilities/Constants.h>
#include <simpla/utilities/macro.h>
#include "Body.h"
#include "GeoObject.h"
#include "ParametricBody.h"
#include "ParametricSurface.h"

#include "Surface.h"

namespace simpla {
namespace geometry {

struct Sphere : public ParametricBody {
    SP_GEO_OBJECT_HEAD(Sphere, ParametricBody)
    Sphere() = default;
    ~Sphere() override = default;

   protected:
    explicit Sphere(Axis const &axis) : ParametricBody(axis) {}
    std::shared_ptr<GeoObject> GetBoundary() const override;

    bool TestIntersection(box_type const &, Real tolerance) const override;
    std::shared_ptr<GeoObject> Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const override;

   public:
    /**
     *
     * @param u R
     * @param v phi
     * @param w theta
     * @return
     */
    point_type xyz(Real r, Real phi, Real theta) const override {
        Real cos_theta = std::cos(theta);
        return m_axis_.Coordinates(r * cos_theta * std::cos(phi), r * cos_theta * std::sin(phi), r * std::sin(theta));
        //        return m_axis_.o + r * std::cos(theta) * (std::cos(phi) * m_axis_.x + std::sin(phi) * m_axis_.y) +
        //               r * std::sin(theta) * m_axis_.z;
    };
    point_type uvw(Real x, Real y, Real z) const override { return point_type{x, y, z}; };
};

struct SphericalSurface : public ParametricSurface {
    SP_GEO_OBJECT_HEAD(SphericalSurface, ParametricSurface);

   protected:
    SphericalSurface() = default;
    SphericalSurface(SphericalSurface const &other) = default;
    SphericalSurface(Axis const &axis, Real radius, Real phi0 = SP_SNaN, Real phi1 = SP_SNaN, Real theta0 = SP_SNaN,
                     Real theta1 = SP_SNaN);

   public:
    ~SphericalSurface() override = default;

    bool IsClosed() const override { return true; };

    void SetRadius(Real r) { m_radius_ = r; }
    Real GetRadius() const { return m_radius_; }

    point_type xyz(Real phi, Real theta) const override {
        Real cos_theta = std::cos(theta);
        return m_axis_.Coordinates(m_radius_ * cos_theta * std::cos(phi), m_radius_ * cos_theta * std::sin(phi),
                                   m_radius_ * std::sin(theta));
        //        return m_axis_.o + m_radius_ * std::cos(theta) * (std::cos(phi) * m_axis_.x + std::sin(phi) *
        //        m_axis_.y) +
        //               m_radius_ * std::sin(theta) * m_axis_.z;
    };
    point_type uvw(Real x, Real y, Real z) const override { return point_type{x, y, z}; };
    bool TestIntersection(point_type const &, Real tolerance) const override;
    bool TestIntersection(box_type const &, Real tolerance) const override;
    std::shared_ptr<GeoObject> Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const override;

   private:
    Real m_radius_ = 1.0;
};
}  // namespace geometry
}  // namespace simpla

#endif  // SIMPLA_SPHERE_H
