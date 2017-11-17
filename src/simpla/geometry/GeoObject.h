/**
 * @file GeoObject.h
 *
 *  Created on: 2015-6-7
 *      Author: salmon
 */

#ifndef CORE_GEOMETRY_GEO_OBJECT_H_
#define CORE_GEOMETRY_GEO_OBJECT_H_
#include <simpla/SIMPLA_config.h>
#include <simpla/data/Serializable.h>
#include "Axis.h"
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
 *   GeoObject<|--Body
 *   Body o-- Surface
 *   Surface o-- Curve
 *   Curve <|-- ParametricCurve
 *   Curve <|-- BoundedCurve
 *   ParametricCurve <|-- Line
 *   ParametricCurve <|-- Conic
 *
 *   BoundedCurve <|-- Polyline
 *   BoundedCurve <|-- BezierCurve
 *   BoundedCurve <|-- BSplineCurve
 *   BoundedCurve <|-- TrimmedCurve
 *
 *   Conic <|-- Circle
 *   Conic <|-- Ellipse
 *   Conic <|-- Hyperbola
 *   Conic <|-- Parabola
 *
 *   Surface <|-- ParametricSurface
 *   ParametricSurface <|-- Plane
 *   ParametricSurface <|-- CylindricalSurface
 *   ParametricSurface <|-- SphericalSurface
 *   ParametricSurface <|-- ToroidalSurface
 *
 *   Surface <|-- BoundedSurface
 *   BoundedSurface <|-- BezierSurface
 *   BoundedSurface <|-- BSplineSurface
 *   BoundedSurface <|-- PatchSurface
 *
 *  ParametricBody o-- ParametricSurface
 *
 *   Body <|-- ParametricBody
 *   ParametricBody <|-- Cube
 *   ParametricBody <|-- Ball
 *   ParametricBody <|-- Cylindrical
 *   ParametricBody <|-- Toroidal
 *
 *   Surface <|-- SweptSurface
 *   SweptSurface <|-- SurfaceOfLinearExtrusion
 *   SweptSurface <|-- SurfaceOfRevolution
 *
 *   Vertex  <|-- GetIntersectionionCurveSurface
 *   Surface <|-- GetIntersectionionSurfaceSolid
 *   Solid   <|-- GetIntersectionionSolidSolid
 *
 *  @enduml
 */
class GeoObject : public std::enable_shared_from_this<GeoObject>, public data::Serializable {
    SP_SERIALIZABLE_HEAD(data::Serializable, GeoObject)

   protected:
    GeoObject();
    GeoObject(GeoObject const &other);
    explicit GeoObject(Axis const &axis);

   public:
    ~GeoObject() override;

    virtual std::shared_ptr<GeoObject> Copy() const = 0;
    std::shared_ptr<const this_type> Self() const { return (shared_from_this()); }
    std::shared_ptr<this_type> Self() { return (shared_from_this()); }

    virtual int GetDimension() const;
    virtual bool IsSimpleConnected() const;
    virtual bool IsConvex() const;
    virtual bool IsContinued() const;
    virtual bool IsClosed() const;

    virtual Axis &GetAxis();
    virtual Axis const &GetAxis() const;
    virtual void SetAxis(Axis const &);

    virtual void Mirror(const point_type &p);
    virtual void Mirror(const Axis &a1);
    virtual void Rotate(const Axis &a1, Real ang);
    virtual void Scale(Real s, int dir);
    virtual void Translate(const vector_type &v);
    virtual void Transform(const Axis &v);
    virtual void Move(const point_type &p);
    void Scale(Real s) { Scale(s, -1); }

    std::shared_ptr<GeoObject> Transformed(const Axis &axis) const;

    virtual std::shared_ptr<GeoObject> GetBoundary() const;
    virtual box_type GetBoundingBox() const;
    virtual bool CheckIntersection(point_type const &x, Real tolerance) const;
    virtual bool CheckIntersection(box_type const &, Real tolerance) const;
    virtual bool CheckIntersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const;
    virtual std::shared_ptr<GeoObject> GetUnion(std::shared_ptr<const GeoObject> const &g, Real tolerance) const;
    virtual std::shared_ptr<GeoObject> GetDifference(std::shared_ptr<const GeoObject> const &g, Real tolerance) const;
    virtual std::shared_ptr<GeoObject> GetIntersection(std::shared_ptr<const GeoObject> const &g, Real tolerance) const;
    std::shared_ptr<GeoObject> GetUnion(std::shared_ptr<const GeoObject> const &g) const {
        return GetUnion(g, SP_GEO_DEFAULT_TOLERANCE);
    }
    std::shared_ptr<GeoObject> GetDifference(std::shared_ptr<const GeoObject> const &g) const {
        return GetDifference(g, SP_GEO_DEFAULT_TOLERANCE);
    }
    std::shared_ptr<GeoObject> GetIntersection(std::shared_ptr<const GeoObject> const &g) const {
        return GetIntersection(g, SP_GEO_DEFAULT_TOLERANCE);
    }

   protected:
    Axis m_axis_;
};

#define SP_GEO_OBJECT_HEAD(_BASE_NAME_, _CLASS_NAME_)                                                                \
    SP_SERIALIZABLE_HEAD(_BASE_NAME_, _CLASS_NAME_)                                                                  \
   protected:                                                                                                        \
    explicit _CLASS_NAME_(Axis const &axis);                                                                         \
    _CLASS_NAME_(_CLASS_NAME_ const &other);                                                                         \
    _CLASS_NAME_();                                                                                                  \
                                                                                                                     \
   public:                                                                                                           \
    ~_CLASS_NAME_() override;                                                                                        \
    std::shared_ptr<GeoObject> Copy() const override { return std::shared_ptr<this_type>(new this_type(*this)); }    \
    std::shared_ptr<this_type> CopyThis() const { return std::dynamic_pointer_cast<this_type>(Copy()); };            \
    std::shared_ptr<_CLASS_NAME_> Self() { return std::dynamic_pointer_cast<this_type>(this->shared_from_this()); }; \
    std::shared_ptr<const _CLASS_NAME_> Self() const {                                                               \
        return std::dynamic_pointer_cast<this_type>(const_cast<this_type *>(this)->shared_from_this());              \
    };                                                                                                               \
                                                                                                                     \
   public:                                                                                                           \
    template <typename... Args>                                                                                      \
    std::shared_ptr<this_type> Mirrored(Args &&... args) const {                                                     \
        auto res = CopyThis();                                                                                       \
        res->Mirror(std::forward<Args>(args)...);                                                                    \
        return res;                                                                                                  \
    }                                                                                                                \
    template <typename... Args>                                                                                      \
    std::shared_ptr<this_type> Rotated(Args &&... args) const {                                                      \
        auto res = CopyThis();                                                                                       \
        res->Rotate(std::forward<Args>(args)...);                                                                    \
        return res;                                                                                                  \
    };                                                                                                               \
    template <typename... Args>                                                                                      \
    std::shared_ptr<this_type> Scaled(Args &&... args) const {                                                       \
        auto res = CopyThis();                                                                                       \
        res->Scale(std::forward<Args>(args)...);                                                                     \
        return res;                                                                                                  \
    }                                                                                                                \
    template <typename... Args>                                                                                      \
    std::shared_ptr<this_type> Translated(Args &&... args) const {                                                   \
        auto res = CopyThis();                                                                                       \
        res->Translate(std::forward<Args>(args)...);                                                                 \
        return res;                                                                                                  \
    }                                                                                                                \
    template <typename... Args>                                                                                      \
    std::shared_ptr<this_type> Moved(Args &&... args) const {                                                        \
        auto res = CopyThis();                                                                                       \
        res->Move(std::forward<Args>(args)...);                                                                      \
        return res;                                                                                                  \
    }

#define SP_GEO_OBJECT_REGISTER(_CLASS_NAME_) \
    bool _CLASS_NAME_::_is_registered =      \
        simpla::Factory<GeoObject>::RegisterCreator<_CLASS_NAME_>(_CLASS_NAME_::RegisterName());

}  // namespace geometry
}  // namespace simpla

#endif /* CORE_GEOMETRY_GEO_OBJECT_H_ */
