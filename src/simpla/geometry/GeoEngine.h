//
// Created by salmon on 17-11-1.
//

#ifndef SIMPLA_ENGINE_H
#define SIMPLA_ENGINE_H

#include <simpla/data/DataNode.h>
#include <simpla/data/SPObject.h>
#include <simpla/utilities/SPDefines.h>
#include <memory>

namespace simpla {
namespace geometry {
class GeoObject;
struct GeoEngine : public std::enable_shared_from_this<GeoEngine> {
   public:
    virtual std::string FancyTypeName() const { return "GeoEngine"; }
    static std::string RegisterName() { return "GeoEngine"; }

   private:
    typedef GeoEngine this_type;

   public:
    GeoEngine();
    virtual ~GeoEngine();

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

#define SP_GEO_ENGINE_HEAD(_CLASS_NAME_, _BASE_NAME_)                                                                \
   public:                                                                                                           \
    std::string FancyTypeName() const override { return base_type::FancyTypeName() + "." + __STRING(_CLASS_NAME_); } \
    static std::string RegisterName() { return __STRING(_CLASS_NAME_); }                                             \
                                                                                                                     \
    static bool _is_registered;                                                                                      \
                                                                                                                     \
   private:                                                                                                          \
    typedef _BASE_NAME_ base_type;                                                                                   \
    typedef _BASE_NAME_##_CLASS_NAME_ this_type;                                                                     \
                                                                                                                     \
   protected:                                                                                                        \
    _BASE_NAME_##_CLASS_NAME_();                                                                                     \
                                                                                                                     \
   public:                                                                                                           \
    ~_BASE_NAME_##_CLASS_NAME_() override;                                                                           \
                                                                                                                     \
   public:                                                                                                           \
    template <typename... Args>                                                                                      \
    static std::shared_ptr<this_type> New(Args &&... args) {                                                         \
        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...));                               \
    }
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_ENGINE_H
