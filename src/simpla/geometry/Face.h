//
// Created by salmon on 17-11-14.
//

#ifndef SIMPLA_FACE_H
#define SIMPLA_FACE_H

#include <simpla/SIMPLA_config.h>
#include "GeoObject.h"
#include "Surface.h"
namespace simpla {
namespace geometry {
struct Face : public GeoObject {
    SP_GEO_OBJECT_HEAD(GeoObject, Face);
    SP_GEO_OBJECT_CREATABLE
    void Deserialize(std::shared_ptr<const simpla::data::DataEntry> const &cfg) override;
    std::shared_ptr<simpla::data::DataEntry> Serialize() const override;

   protected:
    explicit Face(Axis const &axis, std::shared_ptr<const Surface> const &surface = nullptr, Real l = 1, Real w = 1);
    explicit Face(Axis const &axis, std::shared_ptr<const Surface> const &surface, Real u_min, Real u_max, Real v_min,
                  Real v_max);
    explicit Face(Axis const &axis, std::shared_ptr<const Surface> const &surface,
                  std::tuple<point2d_type, point2d_type> const &range);

   public:
    void SetSurface(std::shared_ptr<const Surface> const &s);
    std::shared_ptr<const Surface> GetSurface() const;
    std::tuple<point2d_type, point2d_type> const &GetParameterRange() const;
    void SetParameterRange(std::tuple<point2d_type, point2d_type> const &b);

   private:
    std::shared_ptr<const Surface> m_surface_;
    std::tuple<point2d_type, point2d_type> m_range_{{0, 0}, {1, 1}};
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_FACE_H
