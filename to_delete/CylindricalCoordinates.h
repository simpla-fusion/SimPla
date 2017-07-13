//
// Created by salmon on 16-10-9.
//

#ifndef SIMPLA_CYLINDRICALGEOMETRY_H
#define SIMPLA_CYLINDRICALGEOMETRY_H

#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/algebra.h>
#include <simpla/data/data.h>
#include <simpla/engine/engine.h>
#include <simpla/utilities/FancyStream.h>

#include <iomanip>
#include <vector>
#include "simpla/geometry/Chart.h"
namespace simpla {
namespace geometry {
struct CylindricalCoordinates : public Chart {
    SP_OBJECT_HEAD(CylindricalCoordinates, Chart)

    CylindricalCoordinates() {}

    ~CylindricalCoordinates() override = default;

    SP_DEFAULT_CONSTRUCT(CylindricalCoordinates)
    DECLARE_REGISTER_NAME("CylindricalCoordinates");

    void InitializeData(engine::Mesh *,Real) const;
};
}

}  // namespace simpla

#endif  // SIMPLA_CYLINDRICALGEOMETRY_H
