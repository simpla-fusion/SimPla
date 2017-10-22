/**
 * @file GeoObject.h
 *
 *  Created on: 2015-6-7
 *      Author: salmon
 */

#ifndef CORE_GEOMETRY_GEO_OBJECT_H_
#define CORE_GEOMETRY_GEO_OBJECT_H_
#include "simpla/SIMPLA_config.h"

#include "simpla/data/SPObject.h"

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
 *
 *   GeoObject<|--Vertex
 *   GeoObject<|--Curve
 *   GeoObject<|--Surface
 *   GeoObject<|--Solid
 *
 *   Curve <|-- Line
 *   Curve <|-- Polyline
 *   Curve <|-- Conic
 *   Curve <|-- BoundedCurve
 *
 *   BoundedCurve <|-- BezierCurve
 *   BoundedCurve <|-- BSplineCurve
 *   BoundedCurve <|-- TrimmedCurve
 *
 *   Conic <|-- Circle
 *   Conic <|-- Ellipse
 *   Conic <|-- Hyperbola
 *   Conic <|-- Parabola
 *
 *   Surface <|-- ElementarySurface
 *   ElementarySurface <|-- Plane
 *   ElementarySurface <|-- CylindricalSurface
 *   ElementarySurface <|-- SphericalSurface
 *   ElementarySurface <|-- ToroidalSurface
 *
 *   Surface <|-- BoundedSurface
 *   BoundedSurface <|-- BezierSurface
 *   BoundedSurface <|-- BSplineSurface
 *   BoundedSurface <|-- PatchSurface
 *

 *
 *   Solid <|-- ElementarySolid
 *   ElementarySolid <|-- Cube
 *   ElementarySolid <|-- Ball
 *   ElementarySolid <|-- Cylindrical
 *   ElementarySolid <|-- Toroidal
 *
 *   Surface <|-- SweptSurface
 *   SweptSurface <|-- SurfaceOfLinearExtrusion
 *   SweptSurface <|-- SurfaceOfRevolution
 *
 *   Vertex  <|-- IntersectionCurveSurface
 *   Surface <|-- IntersectionSurfaceSolid
 *   Solid   <|-- IntersectionSolidSolid
 *
 *  @enduml
 */
class GeoObject : public SPObject {
    SP_OBJECT_HEAD(GeoObject, SPObject)
    std::string ClassName() const override { return "GeoObject"; }
    virtual std::shared_ptr<GeoObject> Copy() const { return nullptr; };
    virtual box_type GetBoundingBox() const;
    virtual bool CheckInside(point_type const &x, Real tolerance = SP_DEFAULT_GEOMETRY_TOLERANCE) const {
        return false;
    }

    //    virtual int Dimension() const { return 3; };
    //    virtual Real Measure() const;
    //    virtual box_type GetBoundingBox() const;
    //    virtual std::shared_ptr<GeoObject> GetBoundary() const;
    //    virtual bool CheckInside(point_type const &x, Real tolerance = SP_DEFAULT_GEOMETRY_TOLERANCE) const;
    //    virtual std::shared_ptr<GeoObject> GetBoundary() const { return nullptr; };
    //    /// The axis-aligned minimum bounding box (or AABB) , Cartesian
    //    virtual bool CheckInside(point_type const &x) const;
    //    virtual std::shared_ptr<GeoObject> Intersection(std::shared_ptr<GeoObject> const &other) const;
    //    virtual std::shared_ptr<GeoObject> Difference(std::shared_ptr<GeoObject> const &other) const;
    //    virtual std::shared_ptr<GeoObject> Union(std::shared_ptr<GeoObject> const &other) const;
    //  arbitrarily oriented minimum bounding box  (or OBB)
    //    virtual std::tuple<point_type, vector_type, vector_type, vector_type> OrientedBoundingBox() const;
};

#define SP_GEO_ABS_OBJECT_HEAD(_CLASS_NAME_, _BASE_NAME_)                                 \
   public:                                                                                \
    static std::string FancyTypeName_s() { return __STRING(_CLASS_NAME_); }               \
    virtual std::string FancyTypeName() const override { return __STRING(_CLASS_NAME_); } \
                                                                                          \
   private:                                                                               \
    typedef _BASE_NAME_ base_type;                                                        \
    typedef _CLASS_NAME_ this_type;                                                       \
                                                                                          \
   public:

#define SP_GEO_OBJECT_HEAD(_CLASS_NAME_, _BASE_NAME_)                                     \
   public:                                                                                \
    static std::string FancyTypeName_s() { return __STRING(_CLASS_NAME_); }               \
    virtual std::string FancyTypeName() const override { return __STRING(_CLASS_NAME_); } \
                                                                                          \
    static bool _is_registered;                                                           \
                                                                                          \
   private:                                                                               \
    typedef _BASE_NAME_ base_type;                                                        \
    typedef _CLASS_NAME_ this_type;                                                       \
                                                                                          \
   public:                                                                                \
    void Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) override;        \
    std::shared_ptr<simpla::data::DataNode> Serialize() const override;                   \
                                                                                          \
    template <typename... Args>                                                           \
    static std::shared_ptr<this_type> New(Args &&... args) {                              \
        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...));    \
    };                                                                                    \
    std::shared_ptr<GeoObject> Copy() const override { return std::shared_ptr<this_type>(new this_type(*this)); };

}  // namespace geometry
}  // namespace simpla

#endif /* CORE_GEOMETRY_GEO_OBJECT_H_ */
