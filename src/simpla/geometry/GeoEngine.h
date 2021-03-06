//
// Created by salmon on 17-11-1.
//

#ifndef SIMPLA_ENGINE_H
#define SIMPLA_ENGINE_H

#include <simpla/data/DataEntry.h>
#include <simpla/utilities/SPDefines.h>
#include <simpla/utilities/SingletonHolder.h>
#include <memory>

namespace simpla {
namespace geometry {

void Initialize(std::string const &key = "");
void Initialize(std::shared_ptr<const data::DataEntry> const &);
void Finalize();

class GeoObject;
struct GeoEngineAPI {
   public:
    virtual std::string FancyTypeName() const { return "GeoEngine"; }
    virtual std::string GetRegisterName() const { return RegisterName(); }
    static std::string RegisterName() { return "GeoEngine"; }

   private:
    typedef GeoEngineAPI this_type;

   public:
    GeoEngineAPI();
    virtual ~GeoEngineAPI();
    virtual void Deserialize(std::shared_ptr<simpla::data::DataEntry> const &cfg);
    virtual std::shared_ptr<simpla::data::DataEntry> Serialize() const;
    virtual std::shared_ptr<GeoObject> GetBoundary(std::shared_ptr<const GeoObject> const &) const;
    virtual bool CheckIntersection(std::shared_ptr<const GeoObject> const &, point_type const &x, Real tolerance) const;
    virtual bool CheckIntersection(std::shared_ptr<const GeoObject> const &, box_type const &, Real tolerance) const;
    virtual bool CheckIntersection(std::shared_ptr<const GeoObject> const &, std::shared_ptr<const GeoObject> const &,
                                   Real tolerance) const;

    virtual std::shared_ptr<GeoObject> GetUnion(std::shared_ptr<const GeoObject> const &g0,
                                                std::shared_ptr<const GeoObject> const &g1, Real tolerance) const;
    virtual std::shared_ptr<GeoObject> GetDifference(std::shared_ptr<const GeoObject> const &g0,
                                                     std::shared_ptr<const GeoObject> const &g1, Real tolerance) const;
    virtual std::shared_ptr<GeoObject> GetIntersection(std::shared_ptr<const GeoObject> const &g0,
                                                       std::shared_ptr<const GeoObject> const &g1,
                                                       Real tolerance) const;

    virtual bool IsOpened() const { return m_is_opened_; }
    virtual void OpenFile(std::string const &path);
    virtual void CloseFile();
    virtual void FlushFile();
    virtual std::string GetFilePath() const;
    virtual void Save(std::shared_ptr<const GeoObject> const &geo, std::string const &name) const;
    void Save(std::shared_ptr<const GeoObject> const &geo) const { Save(geo, ""); }
    virtual std::shared_ptr<GeoObject> Load(std::string const &name) const;

   private:
    bool m_is_opened_ = false;
};

#define GEO_ENGINE simpla::SingletonHolder<std::shared_ptr<simpla::geometry::GeoEngineAPI>>::instance()
#define SP_GEO_ENGINE_HEAD(_REGISTER_NAME_)                                             \
   public:                                                                              \
    std::string FancyTypeName() const override {                                        \
        return base_type::FancyTypeName() + "." + __STRING(GeoEngine##_REGISTER_NAME_); \
    }                                                                                   \
    static std::string RegisterName() { return __STRING(_REGISTER_NAME_); }             \
                                                                                        \
    static bool _is_registered;                                                         \
                                                                                        \
   private:                                                                             \
    typedef GeoEngineAPI base_type;                                                     \
    typedef GeoEngine##_REGISTER_NAME_ this_type;                                       \
                                                                                        \
   protected:                                                                           \
    GeoEngine##_REGISTER_NAME_();                                                       \
                                                                                        \
   public:                                                                              \
    ~GeoEngine##_REGISTER_NAME_() override;                                             \
                                                                                        \
   public:                                                                              \
    template <typename... Args>                                                         \
    static std::shared_ptr<this_type> New(Args &&... args) {                            \
        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...));  \
    }
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_ENGINE_H
