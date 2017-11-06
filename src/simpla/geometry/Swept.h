//
// Created by salmon on 17-10-24.
//

#ifndef SIMPLA_SWEPTBODY_H
#define SIMPLA_SWEPTBODY_H
#include <simpla/algebra/nTuple.ext.h>
#include <simpla/utilities/Constants.h>
#include "Body.h"
#include "Curve.h"
#include "PrimitiveShape.h"
#include "Surface.h"
namespace simpla {
namespace geometry {

struct Swept : public PrimitiveShape {
    SP_GEO_ABS_OBJECT_HEAD(Swept, PrimitiveShape);
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_SWEPTBODY_H
