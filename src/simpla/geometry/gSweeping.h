//
// Created by salmon on 17-11-22.
//

#ifndef SIMPLA_GSWEEPING_H
#define SIMPLA_GSWEEPING_H

#include "Axis.h"
#include "GeoEntity.h"
#include "gBody.h"
#include "gCurve.h"
#include "gSurface.h"

namespace simpla {
namespace geometry {

struct gSweeping : public GeoEntity {
    SP_GEO_ENTITY_HEAD(GeoEntity, gSweeping, Sweeping);

    explicit gSweeping(std::shared_ptr<const GeoEntity> const& basis_entity, std::shared_ptr<const gCurve> const& curve,
                       Axis const& r_axis = Axis{});

    void Deserialize(std::shared_ptr<const simpla::data::DataEntry> const& cfg) override;
    std::shared_ptr<simpla::data::DataEntry> Serialize() const override;

    void SetRelativeAxis(Axis const&);
    Axis const& GetRelativeAxis() const;
    void SetPath(std::shared_ptr<const gCurve> const&);
    std::shared_ptr<const gCurve> GetPath() const;
    void SetBasis(std::shared_ptr<const GeoEntity> const&);
    std::shared_ptr<const GeoEntity> GetBasis() const;

    point_type xyz(Real u, Real v, Real w) const override;

   private:
    std::shared_ptr<const gCurve> m_path_;
    std::shared_ptr<const GeoEntity> m_basis_;
    Axis m_r_axis_;
};

}  // namespace geometry {
}  // namespace simpla {
#endif  // SIMPLA_GSWEEPING_H
