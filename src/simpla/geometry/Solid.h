//
// Created by salmon on 17-11-18.
//

#ifndef SIMPLA_SOLID_H
#define SIMPLA_SOLID_H

#include "GeoObject.h"
namespace simpla {
namespace geometry {
struct Body;
struct Solid : public GeoObject {
    SP_GEO_OBJECT_HEAD(GeoObject, Solid);

   protected:
    Solid(Axis const &axis, std::shared_ptr<const Body> const &body, Real u_min, Real u_max, Real v_min, Real v_max,
          Real w_min, Real w_max);
    Solid(Axis const &axis, std::shared_ptr<const Body> const &body, point_type const &u_min, point_type const &u_max);
    Solid(Axis const &axis, std::shared_ptr<const Body> const &body, box_type const &);

   public:
    void SetBody(std::shared_ptr<const Body> const &s) { m_body_ = s; }
    std::shared_ptr<const Body> GetBody() const { return m_body_; }
    void SetParameterRange(point_type const &umin, point_type const &umax) { m_range_ = std::tie(umin, umax); };
    box_type const &GetParameterRange() const { return m_range_; };

   private:
    std::shared_ptr<const Body> m_body_ = nullptr;
    box_type m_range_{{0, 0, 0}, {1, 1, 1}};
};
}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_SOLID_H
