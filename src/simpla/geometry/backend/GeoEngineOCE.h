//
// Created by salmon on 17-11-1.
//

#ifndef SIMPLA_GEOENGINEOCE_H
#define SIMPLA_GEOENGINEOCE_H

#include "../GeoEngine.h"
namespace simpla {
namespace geometry {
struct GeoEngineOCE : public GeoEngine {
    SP_GEO_ENGINE_HEAD(OCE, GeoEngine)
   public:
   protected:
    void SaveAPI(std::shared_ptr<const GeoObject> const &geo, std::string const &path,
                 std::string const &name) const override;
    std::shared_ptr<const GeoObject> LoadAPI(std::string const &path, std::string const &name) const override;

    //    std::shared_ptr<GeoObject> GetBoundaryAPI(std::shared_ptr<const GeoObject> const &) const override;
    bool CheckIntersectionAPI(std::shared_ptr<const GeoObject> const &, point_type const &x,
                              Real tolerance) const override;
    bool CheckIntersectionAPI(std::shared_ptr<const GeoObject> const &, box_type const &,
                              Real tolerance) const override;

    std::shared_ptr<GeoObject> GetUnionAPI(std::shared_ptr<const GeoObject> const &,
                                           std::shared_ptr<const GeoObject> const &g, Real tolerance) const override;
    std::shared_ptr<GeoObject> GetDifferenceAPI(std::shared_ptr<const GeoObject> const &,
                                                std::shared_ptr<const GeoObject> const &g,
                                                Real tolerance) const override;
    std::shared_ptr<GeoObject> GetIntersectionAPI(std::shared_ptr<const GeoObject> const &,
                                                  std::shared_ptr<const GeoObject> const &g,
                                                  Real tolerance) const override;
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_GEOENGINEOCE_H
