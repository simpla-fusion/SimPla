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
    explicit Solid(Axis const &axis, std::shared_ptr<const gBody> const &body,
                   box_type const &range = box_type{{0, 0, 0}, {1, 1, 1}});

   public:
    void SetBody(std::shared_ptr<const gBody> const &s);
    std::shared_ptr<const gBody> GetBody() const;
};
}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_SOLID_H
