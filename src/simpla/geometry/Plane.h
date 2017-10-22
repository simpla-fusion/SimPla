//
// Created by salmon on 17-10-20.
//

#ifndef SIMPLA_PLANE_H
#define SIMPLA_PLANE_H

#include <simpla/SIMPLA_config.h>
#include "Face.h"
#include "GeoObject.h"
namespace simpla {
namespace geometry {
struct Plane : public Face {
    SP_GEO_OBJECT_HEAD(Plane, Face)

    Plane() = default;
    Plane(Plane const&) = default;
    ~Plane() = default;
    Plane(point_type const& v0, point_type const& v1, point_type const& v2) : Plane() { SetVertices(v0, v1, v2); };

   public:
    void SetVertices(point_type const& v0, point_type const& v1, point_type const& v2) {
        m_p[0] = v0;
        m_p[1] = v1;
        m_p[2] = v2;
    }
    nTuple<Real, 3, 3> const& GetVertices() const { return m_p; }

    point_type GetPoint(Real u, Real v) const { return m_p[0] * (1 - u - v) + u * m_p[1] + v * m_p[2]; }

    std::tuple<Real, Real> GetParameter(point_type const& p) const {
        return std::tuple<Real, Real>{dot(p - m_p[0], m_p[1] - m_p[0]), dot(p - m_p[0], m_p[2] - m_p[0])};
    };
    box_type GetBoundingBox() const override;

    //    bool CheckInside(point_type const& x, Real tolerance) const override;
    //    virtual bool CheckInsideUV(Real u, Real v, Real tolerance) const { return true; }

   private:
    nTuple<Real, 3, 3> m_p;
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_PLANE_H
