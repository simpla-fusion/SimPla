//
// Created by salmon on 17-10-24.
//

#ifndef SIMPLA_SWEPTBODY_H
#define SIMPLA_SWEPTBODY_H
//
//#include <simpla/data/Configurable.h>
//#include <simpla/data/Serializable.h>
//#include <simpla/utilities/Constants.h>
//#include "Face.h"
//#include "Shell.h"
//#include "Solid.h"
#include "GeoObject.h"
namespace simpla {
namespace geometry {
// struct Wire;
// struct Edge;
//
// struct Sweep : public Shell {
//    SP_GEO_OBJECT_HEAD(Shell, RevolutionShell);
//
//   protected:
//    explicit RevolutionShell(Axis const &axis, std::shared_ptr<const Wire> const &s, Real min_angle, Real max_angle);
//    explicit RevolutionShell(Axis const &axis, std::shared_ptr<const Wire> const &s, Real angle = TWOPI);
//
//   public:
//    std::shared_ptr<const Wire> GetWire() const { return m_basis_obj_; }
//
//    SP_PROPERTY(Real, MinAngle);
//    SP_PROPERTY(Real, MaxAngle);
//
//   private:
//    std::shared_ptr<const Wire> m_basis_obj_;
//};
// struct RevolutionFace : public Face {
//    SP_GEO_OBJECT_HEAD(Face, RevolutionFace);
//
//   protected:
//    explicit RevolutionFace(Axis const &axis, std::shared_ptr<const Edge> const &s, Real min_angle, Real max_angle);
//    explicit RevolutionFace(Axis const &axis, std::shared_ptr<const Edge> const &s, Real angle = TWOPI);
//
//   public:
//    std::shared_ptr<const Edge> GetEdge() const { return m_basis_obj_; }
//
//    SP_PROPERTY(Real, MinAngle);
//    SP_PROPERTY(Real, MaxAngle);
//
//   private:
//    std::shared_ptr<const Edge> m_basis_obj_;
//};
// struct RevolutionSolid : public Solid {
//    SP_GEO_OBJECT_HEAD(Solid, RevolutionSolid);
//
//   protected:
//    explicit RevolutionSolid(Axis const &axis, std::shared_ptr<const Face> const &s, Real min_angle, Real max_angle);
//    explicit RevolutionSolid(Axis const &axis, std::shared_ptr<const Face> const &s, Real angle = TWOPI);
//
//   public:
//    SP_PROPERTY(Real, MinAngle);
//    SP_PROPERTY(Real, MaxAngle);
//
//    std::shared_ptr<const Face> GetFace() const { return m_basis_obj_; }
//
//   private:
//    std::shared_ptr<const Face> m_basis_obj_;
//};
//
// std::shared_ptr<Shell> MakeRevolution(Axis const &, std::shared_ptr<const Wire> const &, Real angle0, Real angle1);
// std::shared_ptr<Face> MakeRevolution(Axis const &, std::shared_ptr<const Edge> const &, Real angle0, Real angle1);
// std::shared_ptr<Solid> MakeRevolution(Axis const &, std::shared_ptr<const Face> const &, Real angle0, Real angle1);
// template <typename T>
// auto MakeRevolution(Axis const &axis, std::shared_ptr<const T> const &g, Real angle) {
//    return MakeRevolution(axis, g, 0, angle);
//}
//
std::shared_ptr<GeoObject> MakeSweep(std::shared_ptr<const GeoObject> const &face,
                                     std::shared_ptr<const GeoObject> const &c);
}  // namespace simpla
}  // namespace geometry
#endif  // SIMPLA_SWEPTBODY_H
