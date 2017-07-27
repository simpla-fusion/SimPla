/**
* @file csCartesian.h
*
*  Created on: 2015-6-14
*      Author: salmon
*/

#ifndef CORE_GEOMETRY_CS_CARTESIAN_H_
#define CORE_GEOMETRY_CS_CARTESIAN_H_

#include "Chart.h"
#include "simpla/algebra/nTuple.h"
#include "simpla/utilities/macro.h"
#include "simpla/utilities/type_cast.h"
namespace simpla {
namespace geometry {
class Curve;
/**
 * @ingroup  coordinate_system
 * @{
 *  Metric of  Cartesian topology_coordinate system
 */
struct csCartesian : public Chart {
    SP_OBJECT_HEAD(csCartesian, Chart)
   public:
    typedef Real scalar_type;

    SP_DEFAULT_CONSTRUCT(csCartesian);
    std::shared_ptr<Curve> GetAxisCurve(index_tuple const &x, int dir) const override;

    /**
     * metric only diff_scheme the volume of simplex
     *
     */

    point_type map(point_type const &p) const override { return p; }
    point_type inv_map(point_type const &p) const override { return p; }

    Real length(point_type const &p0, point_type const &p1) const override { return std::sqrt(dot(p1 - p0, p1 - p0)); }

    Real area(point_type const &p0, point_type const &p1, point_type const &p2) const override {
        return (std::sqrt(dot(cross(p1 - p0, p2 - p0), cross(p1 - p0, p2 - p0)))) * 0.5;
    }

    Real volume(point_type const &p0, point_type const &p1, point_type const &p2, point_type const &p3) const override {
        return dot(p3 - p0, cross(p1 - p0, p2 - p1)) / 6.0;
    }

    Real area(point_type const &p0, point_type const &p1, int normal) const override {
        return (p1[(normal + 1) % 3] - p0[(normal + 1) % 3]) * (p1[(normal + 2) % 3] - p0[(normal + 2) % 3]);
    };

    Real volume(point_type const &p0, point_type const &p1) const override {
        return (p1[0] - p0[0]) * (p1[1] - p0[1]) * (p1[2] - p0[2]);
    };
};
/** @}*/
}
}  // namespace simpla

#endif /* CORE_GEOMETRY_CS_CARTESIAN_H_ */
