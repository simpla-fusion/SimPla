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
    static std::string FancyTypeName_s() { return "GeoEngine"; }
    virtual std::string FancyTypeName() const { return FancyTypeName_s(); }
    static std::string RegisterName_s();
    virtual std::string RegisterName() const { return RegisterName_s(); }

   private:
    typedef GeoEngine this_type;

   public:
    GeoEngine();
    ~GeoEngine() override;

    virtual void Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg);
    virtual std::shared_ptr<simpla::data::DataNode> Serialize() const;

    static std::shared_ptr<this_type> New(std::string const &key);
    static std::shared_ptr<this_type> New(std::shared_ptr<data::DataNode> const &d);
    static void Initialize(std::string const &d);
    static void Initialize(std::shared_ptr<data::DataNode> const &d = nullptr);
    static void Initialize(int argc, char **argv);
    static void Finalize();
    static GeoEngine &static_entry();
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

#define SP_GEO_ENGINE_HEAD(_CLASS_NAME_, _BASE_NAME_)                                    \
   public:                                                                               \
    static std::string FancyTypeName_s() { return __STRING(_BASE_NAME_##_CLASS_NAME_); } \
    static std::string RegisterName_s() { return __STRING(_CLASS_NAME_); }               \
    std::string FancyTypeName() const override { return FancyTypeName_s(); }             \
    std::string RegisterName() const override { return RegisterName_s(); }               \
                                                                                         \
    static bool _is_registered;                                                          \
                                                                                         \
   private:                                                                              \
    typedef _BASE_NAME_ base_type;                                                       \
    typedef _BASE_NAME_##_CLASS_NAME_ this_type;                                         \
                                                                                         \
   protected:                                                                            \
    _BASE_NAME_##_CLASS_NAME_();                                                         \
                                                                                         \
   public:                                                                               \
    ~_BASE_NAME_##_CLASS_NAME_() override;                                               \
                                                                                         \
   public:                                                                               \
    template <typename... Args>                                                          \
    static std::shared_ptr<this_type> New(Args &&... args) {                             \
        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...));   \
    }
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_ENGINE_H
