//
// Created by salmon on 17-11-18.
//

#ifndef SIMPLA_SOLID_H
#define SIMPLA_SOLID_H

#include "GeoObject.h"
namespace simpla {
namespace geometry {
struct gBody;
struct GeoEntity;
struct Solid : public GeoObjectHandle {
    SP_GEO_OBJECT_HEAD(GeoObjectHandle, Solid);

   protected:
    explicit Solid(std::shared_ptr<const GeoEntity> const &body);
    explicit Solid(std::shared_ptr<const gBody> const &body, Axis const &axis, Real u_min, Real u_max, Real v_min,
                   Real v_max, Real w_min, Real w_max);
    explicit Solid(std::shared_ptr<const gBody> const &body, Axis const &axis, point_type const &u_min,
                   point_type const &u_max);
    explicit Solid(std::shared_ptr<const gBody> const &body, Axis const &axis, box_type const &);

   public:
    void SetBody(std::shared_ptr<const gBody> const &s);
    std::shared_ptr<const gBody> GetBody() const;
};
}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_SOLID_H
