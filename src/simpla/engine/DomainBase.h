//
// Created by salmon on 17-4-5.
//

#ifndef SIMPLA_DOMAINBASE_H
#define SIMPLA_DOMAINBASE_H

#include <simpla/data/all.h>
#include <simpla/model/Chart.h>
#include <simpla/utilities/Signal.h>
#include <memory>
#include "Attribute.h"

namespace simpla {

namespace model {
class GeoObject;
}
namespace engine {
class Patch;
class AttributeGroup;
template <typename TM>
struct EBDomain;
/**
* @brief
*/
class DomainBase : public SPObject, public AttributeGroup, public data::EnableCreateFromDataTable<DomainBase> {
    SP_OBJECT_HEAD(DomainBase, SPObject)
   public:
    using AttributeGroup::attribute_type;

    DomainBase();
    ~DomainBase() override;
    DomainBase(DomainBase const &other);
    DomainBase(DomainBase &&other) noexcept;
    void swap(DomainBase &other);
    DomainBase &operator=(this_type const &other) {
        DomainBase(other).swap(*this);
        return *this;
    }
    DomainBase &operator=(this_type &&other) noexcept {
        DomainBase(other).swap(*this);
        return *this;
    }

    DECLARE_REGISTER_NAME(DomainBase)

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(std::shared_ptr<data::DataTable> const &t) override;

    void Pull(Patch *) override;
    void Push(Patch *) override;

    std::string GetDomainPrefix() const override;

    virtual MeshBase const *GetMesh() const;
    virtual MeshBase const *GetBodyMesh() const;
    virtual MeshBase const *GetBoundaryMesh() const;

    template <typename TL, typename TR>
    void Fill(TL &lhs, TR &&rhs) const {
        FillBody(lhs, std::forward<TR>(rhs));
    };

    template <typename LHS, typename RHS>
    void FillBody(LHS &lhs, RHS &&rhs) const {
//        if (GetBodyMesh() != nullptr) {
//            dynamic_cast<typename LHS::mesh_type const *>(GetBodyMesh())->FillBody(lhs, std::forward<RHS>(rhs));
//        }
    }
    template <typename LHS, typename RHS>
    void FillBoundary(LHS &lhs, RHS &&rhs) const {
//        if (GetBoundaryMesh() != nullptr) {
////            dynamic_cast<EBDomain<typename LHS::mesh_type> const *>(GetBodyMesh())
////                ->FillBody(lhs, std::forward<RHS>(rhs));
//        }
    }

    void SetGeoObject(std::shared_ptr<model::GeoObject> g);
    const model::GeoObject *GetGeoObject() const;

    void DoInitialize() override;
    void DoFinalize() override;
    void DoUpdate() override;
    void DoTearDown() override;

    design_pattern::Signal<void(DomainBase *, Real)> PreInitialCondition;
    design_pattern::Signal<void(DomainBase *, Real)> PostInitialCondition;
    design_pattern::Signal<void(DomainBase *, Real, Real)> PreBoundaryCondition;
    design_pattern::Signal<void(DomainBase *, Real, Real)> PostBoundaryCondition;
    design_pattern::Signal<void(DomainBase *, Real, Real)> PreAdvance;
    design_pattern::Signal<void(DomainBase *, Real, Real)> PostAdvance;

    virtual void DoInitialCondition(Real time_now) {}
    virtual void DoBoundaryCondition(Real time_now, Real dt) {}
    virtual void DoAdvance(Real time_now, Real dt) {}

    void InitialCondition(Real time_now);
    void BoundaryCondition(Real time_now, Real dt);
    void Advance(Real time_now, Real dt);

    void InitialCondition(Patch *, Real time_now);
    void BoundaryCondition(Patch *, Real time_now, Real dt);
    void Advance(Patch *, Real time_now, Real dt);

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

#define DOMAIN_HEAD(_DOMAIN_NAME_, _MESH_TYPE_)                                                  \
   public:                                                                                       \
    template <typename... Args>                                                                  \
    explicit _DOMAIN_NAME_(Args &&... args) : engine::DomainBase(std::forward<Args>(args)...) {} \
    ~_DOMAIN_NAME_() override = default;                                                         \
    SP_DEFAULT_CONSTRUCT(_DOMAIN_NAME_);                                                         \
    std::string GetRegisterName() const override { return RegisterName(); }                      \
    static std::string RegisterName() {                                                          \
        return std::string(__STRING(_DOMAIN_NAME_)) + "." + _MESH_TYPE_::RegisterName();         \
    }                                                                                            \
    static bool is_registered;                                                                   \
    typedef _MESH_TYPE_ mesh_type;

#define DOMAIN_DECLARE_FIELD(_NAME_, _IFORM_) \
    Field<mesh_type, typename mesh_type::scalar_type, _IFORM_> _NAME_{this, "name"_ = __STRING(_NAME_)};
}
}
#endif  // SIMPLA_DOMAINBASE_H
