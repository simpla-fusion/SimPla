//
// Created by salmon on 17-7-27.
//

#ifndef SIMPLA_GEOOBJECTOCC_H
#define SIMPLA_GEOOBJECTOCC_H

#include "../GeoObject.h"

class TopoDS_Shape;
class Bnd_Box;
namespace simpla {
namespace geometry {
struct GeoObjectOCC : public GeoObject {
    SP_OBJECT_HEAD(GeoObjectOCC, GeoObject)

   public:
    void Load(std::string const &);
    void Transform(Real scale, point_type const &location = point_type{0, 0, 0},
                   nTuple<Real, 4> const &rotate = nTuple<Real, 4>{0, 0, 0, 0});
    void DoUpdate();

    TopoDS_Shape const &GetShape() const;
    Bnd_Box const &GetOCCBoundingBox() const;

    box_type BoundingBox() const override;
    bool CheckInside(point_type const &x) const override;
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_GEOOBJECTOCC_H
