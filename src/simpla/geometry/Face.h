//
// Created by salmon on 17-11-14.
//

#ifndef SIMPLA_FACE_H
#define SIMPLA_FACE_H

#include <simpla/SIMPLA_config.h>
#include "GeoObject.h"
#include "gSurface.h"
namespace simpla {
namespace geometry {
struct Face : public GeoObject {
    SP_GEO_OBJECT_HEAD(GeoObject, Face);
    void Deserialize(std::shared_ptr<const simpla::data::DataEntry> const &cfg) override;
    std::shared_ptr<simpla::data::DataEntry> Serialize() const override;

    explicit Face(Axis const &axis, std::shared_ptr<const gSurface> const &surface, Real l = 1, Real w = 1);
    explicit Face(Axis const &axis, std::shared_ptr<const gSurface> const &surface, Real u_min, Real u_max, Real v_min,
                  Real v_max);
    explicit Face(Axis const &axis, std::shared_ptr<const gSurface> const &surface,
                  std::tuple<point2d_type, point2d_type> const &range);

    void SetSurface(std::shared_ptr<const gSurface> const &s);
    std::shared_ptr<const gSurface> GetSurface() const;
    std::tuple<point2d_type, point2d_type> const &GetParameterRange() const;
    void SetParameterRange(std::tuple<point2d_type, point2d_type> const &b);

   private:
    std::shared_ptr<const gSurface> m_surface_;
    std::tuple<point2d_type, point2d_type> m_range_{{0, 0}, {1, 1}};
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_FACE_H
