//
// Created by salmon on 17-7-22.

#ifndef SIMPLA_GEOOBJECTOCC_H
#define SIMPLA_GEOOBJECTOCC_H

#include "../GeoObject.h"

class TopoDS_Shape;
namespace simpla {
namespace geometry {
struct GeometryServiceOCC : public GeometryService {
   public:
    SP_OBJECT_HEAD(GeometryServiceOCC, GeometryService)

    GeometryService();
    explicit GeometryService(GeoObject const &);
    ~GeometryService() override;

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(std::shared_ptr<data::DataTable> const &d) override;

    void Load(std::string const &, std::string const &label = "");
    void DoUpdate() override;

    TopoDS_Shape const *GetShape() const;

    Real measure() const override;
    box_type BoundingBox() const override;
    bool CheckInside(point_type const &x) const override;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_GEOOBJECTOCC_H
