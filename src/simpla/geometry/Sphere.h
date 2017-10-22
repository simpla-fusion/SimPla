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
struct Sphere : public Surface {
    SP_OBJECT_HEAD(Sphere, Surface)
   private:
    Real m_radius_ = 1;
    point_type m_origin_{0, 0, 0};

   protected:
    Sphere(Real r, point_type o) : m_radius_(r), m_origin_(std::move(o)) {}

   public:
    box_type GetBoundingBox() const override {
        box_type b;
        std::get<0>(b) = m_origin_ - m_radius_;
        std::get<1>(b) = m_origin_ + m_radius_;
        return std::move(b);
    };

    bool CheckInside(point_type const &x, Real tolerance) const override {
        return dot((x - m_origin_), (x - m_origin_)) - m_radius_ * m_radius_ < tolerance;
    }
};
}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_SPHERE_H
