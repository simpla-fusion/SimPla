//
// Created by salmon on 17-10-26.
//

#ifndef SIMPLA_POLYPOINTS_H
#define SIMPLA_POLYPOINTS_H

#include "GeoObject.h"
namespace simpla {
namespace geometry {
struct PolyPoint : public GeoObject {
    SP_GEO_OBJECT_HEAD(PolyPoint, GeoObject);

   protected:
    PolyPoint();
    PolyPoint(PolyPoint const &);
    explicit PolyPoint(Axis const &axis);

   public:
    ~PolyPoint() override;

    point_type Value(size_type i) const;
    size_type size() const;
    std::vector<point_type> &data();
    std::vector<point_type> const &data() const;
};

}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_POLYPOINTS_H
