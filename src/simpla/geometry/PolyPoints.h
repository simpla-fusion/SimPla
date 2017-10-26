//
// Created by salmon on 17-10-26.
//

#ifndef SIMPLA_POLYPOINTS_H
#define SIMPLA_POLYPOINTS_H

#include "GeoObject.h"
namespace simpla {
namespace geometry {
struct PolyPoints : public GeoObject {
    SP_GEO_OBJECT_HEAD(PolyPoints, GeoObject);

    std::shared_ptr<const GeoObject> GetBaseShape() const;
    void SetBaseShape(std::shared_ptr<const GeoObject> const &c);
    point_type uvw(size_type i) const;
    point_type xyz(size_type i) const;
    size_type size() const;
};

}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_POLYPOINTS_H
