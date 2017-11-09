//
// Created by salmon on 17-11-2.
//

#ifndef SIMPLA_ITERSECTIONCURVESURFACE_H
#define SIMPLA_ITERSECTIONCURVESURFACE_H

#include <simpla/utilities/Factory.h>
#include "GeoObject.h"
#include "Shape.h"

namespace simpla {
namespace geometry {
struct Curve;
struct Surface;
class IntersectionCurveSurface {
   private:
    typedef IntersectionCurveSurface this_type;

   public:
    virtual std::string FancyTypeName() const { return "IntersectionCurveSurface"; }
    virtual std::string GetRegisterName() const { return RegisterName(); }
    static std::string RegisterName() { return "IntersectionCurveSurface"; }

   protected:
    IntersectionCurveSurface();
    IntersectionCurveSurface(IntersectionCurveSurface const &);
    IntersectionCurveSurface(std::shared_ptr<const Shape> const &g, Real tolerance);

   public:
    virtual ~IntersectionCurveSurface();
//    template <typename... Args>
//    static std::shared_ptr<this_type> New(Args &&... args) {
//        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...));
//    };

    static std::shared_ptr<this_type> Create(std::string const &key = "");
    virtual size_type Intersect(std::shared_ptr<const Curve> const &curve, std::vector<Real> *u) = 0;
    virtual size_type Intersect(std::shared_ptr<const Curve> const &curve, std::vector<Real> *u) const = 0;

    Real GetTolerance() const { return m_tolerance_; }
    void SetTolerance(Real v) { m_tolerance_ = v; }
    std::shared_ptr<const Shape> GetShape() const { return m_shape_; }
    void SetShape(std::shared_ptr<const Shape> const &s) { m_shape_ = s; }

   protected:
    std::shared_ptr<const Shape> m_shape_ = nullptr;
    Real m_tolerance_ = SP_GEO_DEFAULT_TOLERANCE;
};
}  //    namespace geometry{
}  // namespace simpla{

#endif  // SIMPLA_ITERSECTIONCURVESURFACE_H
