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
               public data::EnableCreateFromDataTable<Domain, std::shared_ptr<MeshBase> const &,
                                                      std::shared_ptr<geometry::GeoObject> const &> {
    SP_OBJECT_HEAD(Domain, SPObject)
   public:
    explicit Domain(const std::shared_ptr<MeshBase> &m, const std::shared_ptr<geometry::GeoObject> &g);
    ~Domain() override;
    Domain(Domain const &other);
    Domain(Domain &&other);
    void swap(Domain &other);
    Domain &operator=(this_type const &other) {
        Domain(other).swap(*this);
        return *this;
    }
    Domain &operator=(this_type &&other) {
        Domain(other).swap(*this);
        return *this;
    }

    DECLARE_REGISTER_NAME(Domain)
    std::shared_ptr<DataTable> Serialize() const override;
    void Deserialize(const std::shared_ptr<DataTable> &t) override;

    void Pull(Patch *) override;
    void Push(Patch *t) override;

    std::string GetDomainPrefix() const override;

    MeshBase const *GetMesh() const override;
    MeshBase *GetMesh() override;

    engine::Domain *GetDomain() { return this; }
    engine::Domain const *GetDomain() const { return this; }

    void SetGeoObject(const geometry::GeoObject &g);
    const geometry::GeoObject &GetGeoObject() const;

    void DoInitialize() override;
    void DoFinalize() override;
    void DoUpdate() override;
    void DoTearDown() override;

    //#define DEF_OPERATION(_NAME_, ...)                                                            \
//    virtual void _NAME_(__VA_ARGS__) {}                                                       \
//    design_pattern::Signal<void(this_type *, __VA_ARGS__)> Pre##_NAME_;                       \
//    design_pattern::Signal<void(this_type *, __VA_ARGS__)> Post##_NAME_;                      \
//    template <typename... Args>                                                               \
//    std::shared_ptr<Patch> Do##_NAME_(const std::shared_ptr<Patch> &patch, Args &&... args) { \
//        Deserialize(patch);                                                                          \
//        Pre##_NAME_(std::forward<Args>(args)...);                                             \
//        _NAME_(std::forward<Args>(args)...)                                                   \
//        Post##_NAME_(std::forward<Args>(args)...);                                            \
//        return Serialize();                                                                    \
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

    void DoInitialCondition(Patch *, Real time_now);
    void DoBoundaryCondition(Patch *, Real time_now, Real dt);
    void DoAdvance(Patch *, Real time_now, Real dt);

    template <typename T>
    T GetAttribute(std::string const &k, EntityRange const &r) const {
        return T(AttributeGroup::Get(k)->cast_as<T>(), r);
    };

    template <typename T>
    T GetAttribute(std::string const &k, std::string const &s) const {
        return GetAttribute<T>(k, GetMesh()->GetBodyRange(T::iform, s));
    };
    template <typename T>
    T GetAttribute(std::string const &k) const {
        return T(AttributeGroup::Get(k)->cast_as<T>());
    };

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

#define DOMAIN_HEAD(_DOMAIN_NAME_, _MESH_TYPE_)                                              \
   public:                                                                                   \
    template <typename... Args>                                                              \
    explicit _DOMAIN_NAME_(Args &&... args) : engine::Domain(std::forward<Args>(args)...) {} \
    ~_DOMAIN_NAME_() override = default;                                                     \
    SP_DEFAULT_CONSTRUCT(_DOMAIN_NAME_);                                                     \
    std::string GetRegisterName() const override { return RegisterName(); }                  \
    static std::string RegisterName() {                                                      \
        return std::string(__STRING(_DOMAIN_NAME_)) + "." + _MESH_TYPE_::RegisterName();     \
    }                                                                                        \
    static bool is_registered;                                                               \
    typedef _MESH_TYPE_ mesh_type;

#define DOMAIN_DECLARE_FIELD(_NAME_, _IFORM_) \
    Field<mesh_type, typename mesh_type::scalar_type, _IFORM_> _NAME_{this, "name"_ = __STRING(_NAME_)};
}
}
#endif  // SIMPLA_DOMAIN_H
