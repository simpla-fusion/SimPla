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
    virtual std::string FancyTypeName() const { return FancyTypeName_s(); }
    static std::string FancyTypeName_s() { return "GeoEngine"; }

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
#define SP_GEO_ENGINE__REGISTER(_CLASS_NAME_, _REGISTER_NAME_) \
    bool _CLASS_NAME_::_is_registered =                        \
        simpla::geometry::GeoEngine::RegisterCreator<_CLASS_NAME_>(__STRING(_REGISTER_NAME_));

#define SP_GEO_ENGINE_HEAD(_CLASS_NAME_, _BASE_NAME_)                                                    \
   public:                                                                                               \
    static std::string FancyTypeName_s() { return __STRING(_CLASS_NAME_); }                              \
    std::string FancyTypeName() const override { return simpla::traits::type_name<this_type>::value(); } \
                                                                                                         \
    static bool _is_registered;                                                                          \
                                                                                                         \
   private:                                                                                              \
    typedef _BASE_NAME_ base_type;                                                                       \
    typedef _CLASS_NAME_ this_type;                                                                      \
                                                                                                         \
   protected:                                                                                            \
    _CLASS_NAME_();                                                                                      \
                                                                                                         \
   public:                                                                                               \
    ~_CLASS_NAME_() override;                                                                            \
                                                                                                         \
   public:                                                                                               \
    template <typename... Args>                                                                          \
    static std::shared_ptr<this_type> New(Args &&... args) {                                             \
        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...));                   \
    }
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_ENGINE_H
