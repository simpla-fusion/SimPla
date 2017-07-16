//
// Created by salmon on 17-4-5.
//

#ifndef SIMPLA_DOMAINBASE_H
#define SIMPLA_DOMAINBASE_H

#include "simpla/SIMPLA_config.h"

#include <memory>

#include "simpla/algebra/Array.h"
#include "simpla/algebra/sfc/z_sfc.h"
#include "simpla/data/Data.h"
#include "simpla/geometry/Chart.h"
#include "simpla/utilities/Signal.h"

#include "Attribute.h"
#include "Model.h"

namespace simpla {
namespace engine {
class Patch;
class PatchDataPack;
class AttributeGroup;
class Model;
class MeshBase;

class DomainBase : public SPObject,
                   public AttributeGroup,
                   public data::EnableCreateFromDataTable<DomainBase, MeshBase *, const Model *> {
    SP_OBJECT_HEAD(DomainBase, SPObject)
   public:
    using AttributeGroup::attribute_type;
    DECLARE_REGISTER_NAME(DomainBase)

    DomainBase(MeshBase *m, const Model *model);
    ~DomainBase() override;
    DomainBase(DomainBase const &other);
    DomainBase(DomainBase &&other) noexcept;
    DomainBase &operator=(this_type const &other) = delete;
    DomainBase &operator=(this_type &&other) noexcept = delete;

    virtual const geometry::GeoObject *GetBoundary() const { return m_boundary_; }
    virtual const Model *GetModel() const { return m_model_; }
    virtual MeshBase *GetMesh() { return m_mesh_; }
    virtual const MeshBase *GetMesh() const { return m_mesh_; }

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(std::shared_ptr<data::DataTable> const &t) override;

    void SetRange(std::string const &, Range<EntityId> const &);
    Range<EntityId> &GetRange(std::string const &k);
    Range<EntityId> GetRange(std::string const &k) const;

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

   private:
    MeshBase *m_mesh_;
    engine::Model const *m_model_;
    geometry::GeoObject const *m_boundary_;

};  // class DomainBase

template <typename TM, template <typename> class... Policies>
class Domain : public DomainBase, public Policies<Domain<TM, Policies...>>... {
    typedef Domain<TM, Policies...> domain_type;

    SP_OBJECT_HEAD(domain_type, DomainBase);
    DECLARE_REGISTER_NAME(Domain)

    typedef TM mesh_type;

   public:
    typedef DomainBase::attribute_type attribute_type;

    Domain(MeshBase *msh, const Model *model) : DomainBase(msh, model), Policies<this_type>(this)... {}
    ~Domain() override = default;

    const Model *GetModel() const override { return DomainBase::GetModel(); }
    const mesh_type *GetMesh() const override { return dynamic_cast<mesh_type const *>(DomainBase::GetMesh()); }
    mesh_type *GetMesh() override { return dynamic_cast<mesh_type *>(DomainBase::GetMesh()); }

    Domain(const Domain &) = delete;
    Domain(Domain &&) = delete;
    Domain &operator=(Domain const &) = delete;
    Domain &operator=(Domain &&) = delete;

    void DoInitialCondition(Real time_now) override;
    void DoBoundaryCondition(Real time_now, Real dt) override;
    void DoAdvance(Real time_now, Real dt) override;

    void Deserialize(std::shared_ptr<data::DataTable> const &cfg) override;
    std::shared_ptr<data::DataTable> Serialize() const override;

    template <typename... Args>
    void TryFill(Args &&... args) const;

    template <typename TL, typename TR>
    void FillRange(TL &lhs, TR &&rhs, std::string const &k = "") const;

    template <typename TL, typename TR>
    void FillBody(TL &lhs, TR &&rhs) const {};

    template <typename TL, typename TR>
    void FillBoundary(TL &lhs, TR &&rhs) const {};
};  // class Domain

// template <typename TM, template <typename> class... Policies>
// bool Domain<TM, Policies...>::is_registered = DomainBase::RegisterCreator<Domain<TM, Policies...>>();

#define DEFINE_INVOKE_HELPER(_FUN_NAME_)                                                                           \
    CHECK_MEMBER_FUNCTION(has_mem_fun_##_FUN_NAME_, _FUN_NAME_)                                                    \
    template <typename this_type, typename... Args>                                                                \
    int _invoke_##_FUN_NAME_(std::true_type const &has_function, this_type *self, Args &&... args) {               \
        self->_FUN_NAME_(std::forward<Args>(args)...);                                                             \
        return 1;                                                                                                  \
    }                                                                                                              \
    template <typename this_type, typename... Args>                                                                \
    int _invoke_##_FUN_NAME_(std::false_type const &has_not_function, this_type *self, Args &&... args) {          \
        return 0;                                                                                                  \
    }                                                                                                              \
    template <template <typename> class _T0, typename this_type, typename... Args>                                 \
    int _try_invoke_##_FUN_NAME_(this_type const *self, Args &&... args) {                                         \
        return _invoke_##_FUN_NAME_(has_mem_fun_##_FUN_NAME_<_T0<this_type> const, void, Args...>(),               \
                                    dynamic_cast<_T0<this_type> const *>(self), std::forward<Args>(args)...);      \
    }                                                                                                              \
    template <template <typename> class _T0, typename this_type, typename... Args>                                 \
    int _try_invoke_##_FUN_NAME_(this_type *self, Args &&... args) {                                               \
        return _invoke_##_FUN_NAME_(has_mem_fun_##_FUN_NAME_<_T0<this_type>, void, Args...>(),                     \
                                    dynamic_cast<_T0<this_type> *>(self), std::forward<Args>(args)...);            \
    }                                                                                                              \
    template <template <typename> class _T0, template <typename> class _T1, template <typename> class... _TOthers, \
              typename this_type, typename... Args>                                                                \
    int _try_invoke_##_FUN_NAME_(this_type *self, Args &&... args) {                                               \
        return _try_invoke_##_FUN_NAME_<_T0>(self, std::forward<Args>(args)...) +                                  \
               _try_invoke_##_FUN_NAME_<_T1, _TOthers...>(self, std::forward<Args>(args)...);                      \
    }                                                                                                              \
    template <template <typename> class _T0, template <typename> class _T1, template <typename> class... _TOthers, \
              typename this_type, typename... Args>                                                                \
    int _try_invoke_once_##_FUN_NAME_(this_type *self, Args &&... args) {                                          \
        if (_try_invoke_##_FUN_NAME_<_T0>(self, std::forward<Args>(args)...) == 0) {                               \
            return _try_invoke_##_FUN_NAME_<_T1, _TOthers...>(self, std::forward<Args>(args)...);                  \
        } else {                                                                                                   \
            return 1;                                                                                              \
        }                                                                                                          \
    }

DEFINE_INVOKE_HELPER(InitialCondition)
DEFINE_INVOKE_HELPER(BoundaryCondition)
DEFINE_INVOKE_HELPER(Advance)
DEFINE_INVOKE_HELPER(Deserialize)
DEFINE_INVOKE_HELPER(Serialize)

#undef DEFINE_INVOKE_HELPER

template <typename TM, template <typename> class... Policies>
void Domain<TM, Policies...>::DoInitialCondition(Real time_now) {
    _try_invoke_InitialCondition<Policies...>(this, time_now);
}
template <typename TM, template <typename> class... Policies>
void Domain<TM, Policies...>::DoBoundaryCondition(Real time_now, Real dt) {
    _try_invoke_BoundaryCondition<Policies...>(this, time_now, dt);
}
template <typename TM, template <typename> class... Policies>
void Domain<TM, Policies...>::DoAdvance(Real time_now, Real dt) {
    _try_invoke_Advance<Policies...>(this, time_now, dt);
}
template <typename TM, template <typename> class... Policies>
std::shared_ptr<data::DataTable> Domain<TM, Policies...>::Serialize() const {
    auto res = DomainBase::Serialize();
    _try_invoke_Serialize<Policies...>(this, res.get());
    return res;
};
template <typename TM, template <typename> class... Policies>
void Domain<TM, Policies...>::Deserialize(std::shared_ptr<data::DataTable> const &cfg) {
    _try_invoke_Deserialize<Policies...>(this, cfg);
    DomainBase::Deserialize(cfg);
};
template <typename TM, template <typename> class... Policies>
template <typename LHS, typename RHS>
void Domain<TM, Policies...>::FillRange(LHS &lhs, RHS &&rhs, std::string const &k) const {
    auto r = GetRange(k + "_" + std::to_string(LHS::iform));

    if (r.isNull()) {
        this->Fill(lhs, std::forward<RHS>(rhs), r);
    } else {
        this->Fill(lhs, std::forward<RHS>(rhs));
    }
};

#define DOMAIN_POLICY_HEAD(_NAME_)                   \
   private:                                          \
    typedef THost host_type;                         \
    typedef _NAME_<THost> this_type;                 \
                                                     \
   public:                                           \
    host_type *m_host_ = nullptr;                    \
    _NAME_(host_type *h) noexcept : m_host_(h) {}    \
    virtual ~_NAME_() = default;                     \
    _NAME_(_NAME_ const &other) = delete;            \
    _NAME_(_NAME_ &&other) = delete;                 \
    _NAME_ &operator=(_NAME_ const &other) = delete; \
    _NAME_ &operator=(_NAME_ &&other) = delete;      \
    static std::string RegisterName() { return __STRING(_NAME_); }

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
