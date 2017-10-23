//
// Created by salmon on 17-7-21.
//

#ifndef SIMPLA_SPHERE_H
#define SIMPLA_SPHERE_H

#include "simpla/SIMPLA_config.h"

#include "GeoObject.h"
#include "Surface.h"
namespace simpla {
namespace geometry {
/**
 * A sphere is the surface of a solid ball, here having radius r.
 **/
struct Sphere : public Body {
    SP_GEO_OBJECT_HEAD(Sphere, Body)

   protected:
    Sphere(Real r, point_type o) : m_radius_(r) { m_axis_.SetOrigin(o); }

   public:
    box_type GetBoundingBox() const override {
        box_type b;
        std::get<0>(b) = m_axis_.o - m_radius_;
        std::get<1>(b) = m_axis_.o + m_radius_;
        return std::move(b);
    };

    bool CheckInside(point_type const &x, Real tolerance) const override {
        return dot((x - m_axis_.o), (x - m_axis_.o)) - m_radius_ * m_radius_ < tolerance;
    }

   private:
    Real m_radius_ = 1;
};
}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_SPHERE_H
