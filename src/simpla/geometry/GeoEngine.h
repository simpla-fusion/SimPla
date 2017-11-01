//
// Created by salmon on 17-11-1.
//

#ifndef SIMPLA_ENGINE_H
#define SIMPLA_ENGINE_H

#include <simpla/data/SPObject.h>
#include <simpla/utilities/Factory.h>
namespace simpla {
namespace geometry {
class GeoObject;
struct GeoEngine : public Factory<GeoEngine> {
   public:
    GeoEngine();
    ~GeoEngine() override;

    static void Initialize(std::shared_ptr<data::DataNode> const &d = nullptr);
    static void Initialize(int argc, char **argv);
    static void Finalize();
    static GeoEngine &entry();
    static std::shared_ptr<GeoObject> GetBoundary(std::shared_ptr<const GeoObject> const &);
    static bool CheckIntersection(std::shared_ptr<const GeoObject> const &, point_type const &x, Real tolerance);
    static bool CheckIntersection(std::shared_ptr<const GeoObject> const &, box_type const &, Real tolerance);

    static std::shared_ptr<GeoObject> GetUnion(std::shared_ptr<const GeoObject> const &,
                                               std::shared_ptr<const GeoObject> const &g, Real tolerance);
    static std::shared_ptr<GeoObject> GetDifference(std::shared_ptr<const GeoObject> const &,
                                                    std::shared_ptr<const GeoObject> const &g, Real tolerance);
    static std::shared_ptr<GeoObject> GetIntersection(std::shared_ptr<const GeoObject> const &,
                                                      std::shared_ptr<const GeoObject> const &g, Real tolerance);

   protected:
    virtual std::shared_ptr<GeoObject> GetBoundaryInterface(std::shared_ptr<const GeoObject> const &) const;
    virtual bool CheckIntersectionInterface(std::shared_ptr<const GeoObject> const &, point_type const &x,
                                            Real tolerance) const;
    virtual bool CheckIntersectionInterface(std::shared_ptr<const GeoObject> const &, box_type const &,
                                            Real tolerance) const;

    virtual std::shared_ptr<GeoObject> GetUnionInterface(std::shared_ptr<const GeoObject> const &,
                                                         std::shared_ptr<const GeoObject> const &g,
                                                         Real tolerance) const;
    virtual std::shared_ptr<GeoObject> GetDifferenceInterface(std::shared_ptr<const GeoObject> const &,
                                                              std::shared_ptr<const GeoObject> const &g,
                                                              Real tolerance) const;
    virtual std::shared_ptr<GeoObject> GetIntersectionInterface(std::shared_ptr<const GeoObject> const &,
                                                                std::shared_ptr<const GeoObject> const &g,
                                                                Real tolerance) const;

   private:
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_ENGINE_H
