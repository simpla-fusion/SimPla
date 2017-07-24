//
// Created by salmon on 17-7-24.
//

#ifndef SIMPLA_GEOOBJECTOCC_H
#define SIMPLA_GEOOBJECTOCC_H

#include "GeoObject.h"
namespace simpla {
namespace geometry {
struct GeoObjectOCC : public GeoObject {
   public:
    SP_OBJECT_HEAD(GeoObjectOCC, GeoObject)

    GeoObjectOCC();
    ~GeoObjectOCC() override;

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(std::shared_ptr<data::DataTable> const &d) override;

    void Load(std::string const &);
    void DoUpdate() override;

    box_type BoundingBox() const override;
    bool CheckInside(point_type const &x) const override;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_GEOOBJECTOCC_H
