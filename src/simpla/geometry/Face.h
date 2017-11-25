//
// Created by salmon on 17-11-14.
//

#ifndef SIMPLA_FACE_H
#define SIMPLA_FACE_H

#include <simpla/SIMPLA_config.h>
#include "GeoObject.h"
#include "gSurface.h"
namespace simpla {
namespace geometry {
struct Face : public GeoObjectHandle {
    SP_GEO_OBJECT_HEAD(GeoObjectHandle, Face);

   protected:
    explicit Face(Axis const &axis, std::shared_ptr<const gSurface> const &surface,
                  std::tuple<point2d_type, point2d_type> const &range = {{0, 0}, {1, 1}});

   public:
    void SetSurface(std::shared_ptr<const gSurface> const &s);
    std::shared_ptr<const gSurface> GetSurface() const;
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_FACE_H
