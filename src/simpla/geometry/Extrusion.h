//
// Created by salmon on 17-10-24.
//

#ifndef SIMPLA_LINEAREXTRUSIONBODY_H
#define SIMPLA_LINEAREXTRUSIONBODY_H

#include <simpla/utilities/Constants.h>
#include "Swept.h"
namespace simpla {
namespace geometry {
struct Curve;
struct Extrusion : public Swept {
    SP_GEO_OBJECT_HEAD(Extrusion, Swept);

   protected:
    Extrusion();
    Extrusion(Extrusion const &other);
    Extrusion(std::shared_ptr<const Surface> const &s, vector_type const &c);

   protected:
   public:
    ~Extrusion() override;
};
struct ExtrusionSurface : public SweptSurface {
    SP_GEO_OBJECT_HEAD(ExtrusionSurface, SweptSurface);

   protected:
    ExtrusionSurface();
    ExtrusionSurface(ExtrusionSurface const &other);
    ExtrusionSurface(Axis const &axis);

   public:
    ~ExtrusionSurface() override;
    bool IsClosed() const override;
};

}  // namespace simpla
}  // namespace geometry
#endif  // SIMPLA_LINEAREXTRUSIONBODY_H
