//
// Created by salmon on 17-11-14.
//

#ifndef SIMPLA_FACE_H
#define SIMPLA_FACE_H

#include <simpla/SIMPLA_config.h>
#include "GeoObject.h"
#include "PrimitiveShape.h"
#include "Surface.h"
namespace simpla {
namespace geometry {
struct Face : public GeoObject {
    SP_GEO_OBJECT_HEAD(Face, GeoObject);

   protected:
    Face(std::shared_ptr<const Surface> const &surface, Real l, Real w);
    Face(std::shared_ptr<const Surface> const &surface, Real u_min, Real u_max, Real v_min, Real v_max);

   public:
    void SetSurface(std::shared_ptr<const Surface> const &s) { m_surface_ = s; }
    std::shared_ptr<const Surface> GetSurface() const { return m_surface_; }
    std::tuple<point2d_type, point2d_type> const &GetUVRange() const { return m_range_; };
    void SetUVRange(std::tuple<point2d_type, point2d_type> const &b) { m_range_ = b; }

    virtual point_type xy(Real u, Real v) const;
    virtual point_type uv(Real x, Real y) const;

   private:
    std::shared_ptr<const Surface> m_surface_;
    std::tuple<point2d_type, point2d_type> m_range_{{0, 0}, {1, 1}};
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_FACE_H
