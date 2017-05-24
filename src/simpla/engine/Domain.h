//
// Created by salmon on 17-4-5.
//

#ifndef SIMPLA_DOMAIN_H
#define SIMPLA_DOMAIN_H

#include <simpla/data/all.h>
#include <simpla/geometry/GeoObject.h>
#include <simpla/utilities/Signal.h>
#include <memory>
#include "Attribute.h"
#include "MeshBase.h"
namespace simpla {
namespace engine {
class MeshBase;
class Patch;
class AttributeGroup;

/**
* @brief
*/
class Domain : public SPObject,
               public AttributeGroup,
               public data::EnableCreateFromDataTable<Domain, std::shared_ptr<geometry::GeoObject>> {
    SP_OBJECT_HEAD(Domain, SPObject)
   public:
    explicit Domain(std::shared_ptr<geometry::GeoObject> const &g);
    ~Domain() override;

    SP_DEFAULT_CONSTRUCT(Domain);
    DECLARE_REGISTER_NAME("Domain")

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(const std::shared_ptr<data::DataTable> &t) override;

    virtual MeshBase *GetMesh() = 0;
    virtual MeshBase const *GetMesh() const = 0;

    engine::Domain *GetDomain() { return this; }
    engine::Domain const *GetDomain() const { return this; }

    void AddGeoObject(std::string const &k, std::shared_ptr<geometry::GeoObject> const &g);
    std::shared_ptr<geometry::GeoObject> GetGeoObject(std::string const &k = "") const;
    EntityRange GetRange(std::string const &k = "") const;
    EntityRange GetBodyRange(int IFORM = VERTEX, std::string const &k = "") const;
    EntityRange GetBoundaryRange(int IFORM = VERTEX, std::string const &k = "", bool is_parallel = true) const;
    EntityRange GetParallelBoundaryRange(int IFORM = VERTEX, std::string const &k = "") const;
    EntityRange GetPerpendicularBoundaryRange(int IFORM = VERTEX, std::string const &k = "") const;

    EntityRange GetInnerRange(int IFORM = VERTEX) const;
    EntityRange GetGhostRange(int IFORM = VERTEX) const;

    void Initialize() override;
    void Finalize() override;
    void SetUp() override;
    void TearDown() override;

    void Push(const std::shared_ptr<Patch> &);
    std::shared_ptr<Patch> PopPatch();

//#define DEF_OPERATION(_NAME_, ...)                                                            \
//    virtual void _NAME_(__VA_ARGS__) {}                                                       \
//    design_pattern::Signal<void(this_type *, __VA_ARGS__)> Pre##_NAME_;                       \
//    design_pattern::Signal<void(this_type *, __VA_ARGS__)> Post##_NAME_;                      \
//    template <typename... Args>                                                               \
//    std::shared_ptr<Patch> Do##_NAME_(const std::shared_ptr<Patch> &patch, Args &&... args) { \
//        Push(patch);                                                                          \
//        Pre##_NAME_(std::forward<Args>(args)...);                                             \
//        _NAME_(std::forward<Args>(args)...)                                                   \
//        Post##_NAME_(std::forward<Args>(args)...);                                            \
//        return PopPatch();                                                                    \
//    };

    design_pattern::Signal<void(Domain *, Real)> PreInitialCondition;
    design_pattern::Signal<void(Domain *, Real)> PostInitialCondition;
    design_pattern::Signal<void(Domain *, Real, Real)> PreBoundaryCondition;
    design_pattern::Signal<void(Domain *, Real, Real)> PostBoundaryCondition;
    design_pattern::Signal<void(Domain *, Real, Real)> PreAdvance;
    design_pattern::Signal<void(Domain *, Real, Real)> PostAdvance;

    virtual void InitialCondition(Real time_now) {}
    virtual void BoundaryCondition(Real time_now, Real dt) {}
    virtual void Advance(Real time_now, Real dt) {}

    std::shared_ptr<Patch> DoInitialCondition(const std::shared_ptr<Patch> &, Real time_now);
    std::shared_ptr<Patch> DoBoundaryCondition(const std::shared_ptr<Patch> &, Real time_now, Real dt);
    std::shared_ptr<Patch> DoAdvance(const std::shared_ptr<Patch> &, Real time_now, Real dt);

    template <typename T>
    T GetAttribute(std::string const &k, EntityRange const &r) const {
        return T(Get(k)->cast_as<T>(), r);
    };

    template <typename T>
    T GetAttribute(std::string const &k, std::string const &s = "") const {
        return GetAttribute<T>(k, GetBodyRange(T::iform, s));
    };

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

#define DOMAIN_HEAD(_DOMAIN_NAME_, _BASE_TYPE_, _MESH_TYPE_)                                                           \
   public:                                                                                                             \
    explicit _DOMAIN_NAME_(std::shared_ptr<geometry::GeoObject> const &g = nullptr) : _BASE_TYPE_(g), m_mesh_(this) {} \
    ~_DOMAIN_NAME_() override = default;                                                                               \
    SP_DEFAULT_CONSTRUCT(_DOMAIN_NAME_);                                                                               \
    DECLARE_REGISTER_NAME(std::string(__STRING(_DOMAIN_NAME_)) + "<" + _MESH_TYPE_::RegisterName() + ">")              \
    typedef _MESH_TYPE_ mesh_type;                                                                                     \
    template <int IFORM, int DOF = 1>                                                                                  \
    using field_type = Field<mesh_type, typename _MESH_TYPE_::scalar_type, IFORM, DOF>;                                \
    mesh_type m_mesh_;                                                                                                 \
    MeshBase *GetMesh() override { return &m_mesh_; }                                                                  \
    MeshBase const *GetMesh() const override { return &m_mesh_; }

#define DOMAIN_DECLARE_FIELD(_NAME_, _IFORM_, _DOF_, ...)                                                      \
    Field<mesh_type, typename mesh_type::scalar_type, _IFORM_, _DOF_> _NAME_{this, "name"_ = __STRING(_NAME_), \
                                                                             __VA_ARGS__};
}
}
#endif  // SIMPLA_DOMAIN_H
