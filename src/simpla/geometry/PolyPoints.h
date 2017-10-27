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

   protected:
    PolyPoints();
    PolyPoints(PolyPoints const &);
    explicit PolyPoints(Axis const &axis);

   public:
    ~PolyPoints() override;

    point_type Value(size_type i) const;
    size_type size() const;
    std::vector<point_type> &data();
    std::vector<point_type> const &data() const;
};

}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_POLYPOINTS_H
