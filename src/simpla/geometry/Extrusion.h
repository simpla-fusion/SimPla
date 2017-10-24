//
// Created by salmon on 17-10-24.
//

#ifndef SIMPLA_LINEAREXTRUSIONBODY_H
#define SIMPLA_LINEAREXTRUSIONBODY_H

#include <simpla/utilities/Constants.h>
#include "SweptBody.h"
namespace simpla {
namespace geometry {
struct Curve;
struct Extrusion : public SweptBody {
    SP_GEO_OBJECT_HEAD(Extrusion, SweptBody);

   protected:
    Extrusion();
    Extrusion(Extrusion const &other);
    Extrusion(std::shared_ptr<const Surface> const &s, vector_type const &c);

   public:
    ~Extrusion() override;
};

}  // namespace simpla
}  // namespace geometry
#endif  // SIMPLA_LINEAREXTRUSIONBODY_H
