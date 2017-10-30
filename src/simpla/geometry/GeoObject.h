/**
 * @file GeoObject.h
 *
 *  Created on: 2015-6-7
 *      Author: salmon
 */

#ifndef CORE_GEOMETRY_GEO_OBJECT_H_
#define CORE_GEOMETRY_GEO_OBJECT_H_
#include "simpla/SIMPLA_config.h"

#include "Axis.h"
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
   private:
    typedef GeoObject this_type;
    typedef SPObject base_type;

   public:
    GeoObject();
    GeoObject(GeoObject const &other);
    ~GeoObject() override;
    explicit GeoObject(Axis const &axis);
    virtual std::shared_ptr<GeoObject> Copy() const = 0;
    static std::shared_ptr<this_type> New(std::shared_ptr<data::DataNode> const &cfg) {
        return std::dynamic_pointer_cast<this_type>(simpla::SPObject::Create(cfg));
    };
    std::shared_ptr<data::DataNode> Serialize() const override;
    void Deserialize(std::shared_ptr<data::DataNode> const &) override;

    static std::string FancyTypeName_s() { return "GeoObject"; }
    std::string FancyTypeName() const override { return simpla::traits::type_name<this_type>::value(); }
    std::string ClassName() const override { return "GeoObject"; }

    virtual std::shared_ptr<GeoObject> GetBoundary() const;
    virtual box_type GetBoundingBox() const;
    virtual Axis &GetAxis();
    virtual Axis const &GetAxis() const;
    virtual void SetAxis(Axis const &);
    /**
    * @return
    *  <  0 no overlap
    *  == 0 partial overlap
    *  >  1 all inside
    */
    virtual bool TestIntersection(box_type const &) const;
    virtual bool TestInside(point_type const &x) const;
    bool TestInside(Real x, Real y = 0, Real z = 0) const;
    virtual bool TestInsideUVW(point_type const &x) const;
    bool TestInsideUVW(Real u, Real v = 0, Real w = 0) const;

    virtual point_type Value(point_type const &x) const;

    virtual std::shared_ptr<GeoObject> Intersection(std::shared_ptr<const GeoObject> const &g, Real tolerance) const;

    virtual void Mirror(const point_type &p);
    virtual void Mirror(const Axis &a1);
    virtual void Rotate(const Axis &a1, Real ang);
    virtual void Scale(Real s, int dir);
    virtual void Translate(const vector_type &v);
    virtual void Move(const point_type &p);
    void Scale(Real s) { Scale(s, -1); }

   protected:
    Axis m_axis_{};
};

#define SP_GEO_ABS_OBJECT_HEAD(_CLASS_NAME_, _BASE_NAME_)                                                 \
   public:                                                                                                \
    static std::string FancyTypeName_s() { return __STRING(_CLASS_NAME_); }                               \
    virtual std::string FancyTypeName() const override { return __STRING(_CLASS_NAME_); }                 \
                                                                                                          \
   private:                                                                                               \
    typedef _BASE_NAME_ base_type;                                                                        \
    typedef _CLASS_NAME_ this_type;                                                                       \
                                                                                                          \
   public:                                                                                                \
    void Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) override;                        \
    std::shared_ptr<simpla::data::DataNode> Serialize() const override;                                   \
    std::shared_ptr<this_type> CopyThis() const { return std::dynamic_pointer_cast<this_type>(Copy()); }; \
    template <typename... Args>                                                                           \
    std::shared_ptr<this_type> Mirrored(Args &&... args) const {                                          \
        auto res = CopyThis();                                                                            \
        res->Mirror(std::forward<Args>(args)...);                                                         \
        return res;                                                                                       \
    }                                                                                                     \
    template <typename... Args>                                                                           \
    std::shared_ptr<this_type> Rotated(Args &&... args) const {                                           \
        auto res = CopyThis();                                                                            \
        res->Rotate(std::forward<Args>(args)...);                                                         \
        return res;                                                                                       \
    };                                                                                                    \
    template <typename... Args>                                                                           \
    std::shared_ptr<this_type> Scaled(Args &&... args) const {                                            \
        auto res = CopyThis();                                                                            \
        res->Scale(std::forward<Args>(args)...);                                                          \
        return res;                                                                                       \
    }                                                                                                     \
    template <typename... Args>                                                                           \
    std::shared_ptr<this_type> Translated(Args &&... args) const {                                        \
        auto res = CopyThis();                                                                            \
        res->Translate(std::forward<Args>(args)...);                                                      \
        return res;                                                                                       \
    }                                                                                                     \
    template <typename... Args>                                                                           \
    std::shared_ptr<this_type> Moved(Args &&... args) const {                                             \
        auto res = CopyThis();                                                                            \
        res->Move(std::forward<Args>(args)...);                                                           \
        return res;                                                                                       \
    }

#define SP_GEO_OBJECT_HEAD(_CLASS_NAME_, _BASE_NAME_)                                     \
   public:                                                                                \
    static std::string FancyTypeName_s() { return __STRING(_CLASS_NAME_); }               \
    virtual std::string FancyTypeName() const override { return __STRING(_CLASS_NAME_); } \
                                                                                          \
   private:                                                                               \
    typedef _BASE_NAME_ base_type;                                                        \
    typedef _CLASS_NAME_ this_type;                                                       \
    static bool _is_registered;                                                           \
                                                                                          \
   public:                                                                                \
    void Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) override;        \
    std::shared_ptr<simpla::data::DataNode> Serialize() const override;                   \
                                                                                          \
    template <typename... Args>                                                           \
    static std::shared_ptr<this_type> New(Args &&... args) {                              \
        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...));    \
    };                                                                                    \
    static std::shared_ptr<this_type> New(std::shared_ptr<data::DataNode> const &cfg) {   \
        return std::dynamic_pointer_cast<this_type>(simpla::SPObject::Create(cfg));       \
    };                                                                                    \
                                                                                          \
    std::shared_ptr<GeoObject> Copy() const override { return std::shared_ptr<this_type>(new this_type(*this)); };

}  // namespace geometry
}  // namespace simpla

#endif /* CORE_GEOMETRY_GEO_OBJECT_H_ */
