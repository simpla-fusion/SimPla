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
    Extrusion(std::shared_ptr<const Surface> const &s, vector_type const &c);
};

}  // namespace simpla
}  // namespace geometry
#endif  // SIMPLA_LINEAREXTRUSIONBODY_H
