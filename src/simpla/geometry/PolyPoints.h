//
// Created by salmon on 17-10-26.
//

#ifndef SIMPLA_POLYPOINTS_H
#define SIMPLA_POLYPOINTS_H

#include "GeoObject.h"
namespace simpla {
namespace geometry {
struct PolyPoints : public GeoObject {
    SP_GEO_OBJECT_HEAD(GeoObject, PolyPoints);

   public:
    virtual size_type size() const { return 0; };
    virtual point_type GetPoint(size_type idx) const { return point_type{0, 0, 0}; };

    //    virtual point_type Value(size_type i) const = 0;
    //    virtual size_type size() const = 0;
    //
    //    std::shared_ptr<GeoObject> GetBoundary() const override;
    //    box_type GetBoundingBox() const override;
    //    bool CheckIntersection(box_type const &) const override;
    //    bool TestInside(point_type const &x, Real tolerance) const override;
    //    bool TestInsideUVW(point_type const &x, Real tolerance) const override;
    //    std::shared_ptr<GeoObject> GetIntersectionion(std::shared_ptr<const GeoObject> const &g, Real tolerance) const
    //    override;
    //
    //    point_type Value(point_type const &x) const override;
};

}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_POLYPOINTS_H
