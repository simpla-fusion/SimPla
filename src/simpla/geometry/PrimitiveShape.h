//
// Created by salmon on 17-11-6.
//

#ifndef SIMPLA_PRIMITIVESSHAPE_H
#define SIMPLA_PRIMITIVESSHAPE_H
#include <simpla/data/DataEntry.h>
#include <simpla/utilities/SPDefines.h>
#include <memory>
#include "Shape.h"
namespace simpla {
namespace geometry {
struct Body;
struct Shell;
struct PrimitiveShape : public Shape {
    SP_FACTORY_HEAD(Shape, PrimitiveShape, Shape)

   public:
    virtual point_type xyz(Real r, Real phi, Real theta) const;
    virtual point_type uvw(Real x, Real y, Real z) const;

   protected:
};
}  // namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_PRIMITIVESSHAPE_H
