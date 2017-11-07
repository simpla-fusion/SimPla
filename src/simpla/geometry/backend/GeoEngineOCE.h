//
// Created by salmon on 17-11-1.
//

#ifndef SIMPLA_GEOENGINEOCE_H
#define SIMPLA_GEOENGINEOCE_H

#include "../GeoEngine.h"
namespace simpla {
namespace geometry {
struct GeoEngineOCE : public GeoEngineAPI {
   public:
    std::string FancyTypeName() const override { return base_type::FancyTypeName() + ".GeoEngineOCE"; }
    static std::string RegisterName() { return "OCE"; }
    static int _is_registered;

   private:
    typedef GeoEngineAPI base_type;
    typedef GeoEngineOCE this_type;

   protected:
    GeoEngineOCE();

   public:
    ~GeoEngineOCE() override;

   public:
    template <typename... Args>
    static std::shared_ptr<this_type> New(Args &&... args) {
        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...));
    }
    void Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) override;
    std::shared_ptr<simpla::data::DataNode> Serialize() const override;

    void OpenFile(std::string const &path) override;
    void CloseFile() override;
    void DumpFile() override;
    std::string GetFilePath() const override;
    void Save(std::shared_ptr<const GeoObject> const &geo, std::string const &name) const override;
    std::shared_ptr<GeoObject> Load(std::string const &name) const override;

    //    std::shared_ptr<GeoObject> GetBoundary(std::shared_ptr<const GeoObject> const &) const override;
    bool CheckIntersection(std::shared_ptr<const GeoObject> const &, point_type const &x,
                           Real tolerance) const override;
    bool CheckIntersection(std::shared_ptr<const GeoObject> const &, box_type const &, Real tolerance) const override;

    std::shared_ptr<GeoObject> GetUnion(std::shared_ptr<const GeoObject> const &,
                                        std::shared_ptr<const GeoObject> const &g, Real tolerance) const override;
    std::shared_ptr<GeoObject> GetDifference(std::shared_ptr<const GeoObject> const &,
                                             std::shared_ptr<const GeoObject> const &g, Real tolerance) const override;
    std::shared_ptr<GeoObject> GetIntersection(std::shared_ptr<const GeoObject> const &,
                                               std::shared_ptr<const GeoObject> const &g,
                                               Real tolerance) const override;

   private:
    struct pimpl_s;
    pimpl_s *m_pimpl_ = nullptr;
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_GEOENGINEOCE_H
