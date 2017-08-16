/**
 * @file GeoObject.h
 *
 *  Created on: 2015-6-7
 *      Author: salmon
 */

#ifndef CORE_GEOMETRY_GEO_OBJECT_H_
#define CORE_GEOMETRY_GEO_OBJECT_H_

#include "simpla/SIMPLA_config.h"

#include "simpla/algebra/nTuple.ext.h"
#include "simpla/algebra/nTuple.h"
#include "simpla/engine/SPObject.h"
#include "simpla/utilities/Factory.h"
#include "simpla/utilities/Log.h"
#include "simpla/utilities/SPDefines.h"
#include "simpla/utilities/type_traits.h"

#include "BoxUtilities.h"

namespace simpla {
namespace geometry {

/**
 * @ingroup geometry
 *
 *  Abstract class GeoObject is the root class of all geometric objects.
 *
 *  @startuml
 *  class GeoObject{
 *  }
 *
 *  class Transform <TGeo0,TGeo1>{
 *  }
 *
 *   GeoObject<|--Vertex
 *   GeoObject<|--Curve
 *   GeoObject<|--Surface
 *   GeoObject<|--Solid
 *
 *   Curve <|-- Line
 *   Curve <|-- Conic
 *   Curve <|-- BoundedCurve
 *
 *   BoundedCurve <|-- BezierCurve
 *   BoundedCurve <|-- BSplineCurve
 *   BoundedCurve <|-- TrimmedCurve
 *
 *   Conic <|-- Circle/Arc
 *   Conic <|-- Ellipse
 *   Conic <|-- Hyperbola
 *   Conic <|-- Parabola
 *
 *   Surface <|-- Plane
 *
 *   Solid <|-- ElementarySolid
 *   ElementarySolid <|-- Cube
 *   ElementarySolid <|-- Sphere
 *
 *   Surface <|-- TransformCurveCurve
 *   Solid   <|-- TransformSurfaceCurve
 *   Vertex  <|-- IntersectionCurveSurface
 *   Surface <|-- IntersectionSurfaceSolid
 *   Solid   <|-- IntersectionSolidSolid
 *
 *  @enduml
 */
class GeoObject : public engine::SPObject {
    SP_OBJECT_DECLARE_MEMBERS(GeoObject, engine::SPObject)

    static std::shared_ptr<GeoObject> New(std::shared_ptr<const data::DataEntity> const &);

    virtual int Dimension() const { return 3; };
    virtual Real Measure() const;
    virtual std::shared_ptr<GeoObject> Boundary() const { return nullptr; };
    /// The axis-aligned minimum bounding box (or AABB) , Cartesian
    virtual box_type BoundingBox() const;
    virtual bool CheckInside(point_type const &x) const;

    /// arbitrarily oriented minimum bounding box  (or OBB)
    //    virtual std::tuple<point_type, vector_type, vector_type, vector_type> OrientedBoundingBox() const;
};

}  // namespace geometry
}  // namespace simpla

#endif /* CORE_GEOMETRY_GEO_OBJECT_H_ */
