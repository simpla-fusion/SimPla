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

class DomainBase : public SPObject,
                   public AttributeGroup,
                   public data::EnableCreateFromDataTable<DomainBase, std::string, const geometry::Chart *> {
    SP_OBJECT_HEAD(DomainBase, SPObject)
    DECLARE_REGISTER_NAME(DomainBase)
   public:
    using AttributeGroup::attribute_type;

    DomainBase(std::string const &s_name, const geometry::Chart *);
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
    const geometry::Chart *GetChart() const { return m_chart_; }

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(std::shared_ptr<data::DataTable> const &t) override;

    void SetRange(std::string const &, Range<EntityId> const &);
    virtual Range<EntityId> &GetRange(std::string const &k);
    virtual Range <EntityId> GetRange(std::string const &k) const;

    void SetBlock(const MeshBlock &blk);
    virtual const MeshBlock &GetBlock() const;
    virtual id_type GetBlockId() const;

    const Model &GetModel() const { return *m_model_; }
    Model &GetModel() { return *m_model_; }

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

    void Pull(Patch *) override;
    void Push(Patch *) override;

    void InitialCondition(Patch *, Real time_now);
    void BoundaryCondition(Patch *, Real time_now, Real dt);
    void Advance(Patch *, Real time_now, Real dt);

   private:
    MeshBlock m_mesh_block_;

    const geometry::Chart *m_chart_;
    std::shared_ptr<engine::Model> m_model_;

    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;

};  // class DomainBase

template <template <typename> class... Policies>
class Domain : public DomainBase, public Policies<Domain<Policies...>>... {
    typedef Domain<Policies...> host_type;

    SP_OBJECT_HEAD(host_type, DomainBase);

   public:
    typedef DomainBase::attribute_type attribute_type;

    Domain(std::string const &s_name, geometry::Chart const *c) : DomainBase(s_name, c), Policies<this_type>(this)... {}
    ~Domain() override = default;

    static bool is_registered;
    std::string GetRegisterName() const override { return RegisterName(); }

   private:
    template <template <typename> class _T0>
    static std::string _RegisterName() {
        return _T0<this_type>::RegisterName();
    }

    template <template <typename> class _T0, template <typename> class _T1, template <typename> class... _TOthers>
    static std::string _RegisterName() {
        return _T0<this_type>::RegisterName() + "," + _RegisterName<_T1, _TOthers...>();
    }

   public:
    static std::string RegisterName() { return "Domain<" + _RegisterName<Policies...>() + ">"; }

    const geometry::Chart *GetChart() const override { return DomainBase::GetChart(); };
    const engine::MeshBlock &GetBlock() const override { return DomainBase::GetBlock(); };

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

//    template <typename TL, typename TR>
//    void FillBody(TL &lhs, TR &&rhs) const;
//
//    template <typename TL, typename TR>
//    void FillBoundary(TL &lhs, TR &&rhs) const;
};  // class Domain

template <template <typename> class... Policies>
bool Domain<Policies...>::is_registered = DomainBase::RegisterCreator<Domain<Policies...>>();

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

template <template <typename> class... Policies>
void Domain<Policies...>::DoInitialCondition(Real time_now) {
    _try_invoke_InitialCondition<Policies...>(this, time_now);
}
template <template <typename> class... Policies>
void Domain<Policies...>::DoBoundaryCondition(Real time_now, Real dt) {
    _try_invoke_BoundaryCondition<Policies...>(this, time_now, dt);
}
template <template <typename> class... Policies>
void Domain<Policies...>::DoAdvance(Real time_now, Real dt) {
    _try_invoke_Advance<Policies...>(this, time_now, dt);
}
template <template <typename> class... Policies>
std::shared_ptr<data::DataTable> Domain<Policies...>::Serialize() const {
    auto res = DomainBase::Serialize();
    _try_invoke_Serialize<Policies...>(this, res.get());
    return res;
};
template <template <typename> class... Policies>
void Domain<Policies...>::Deserialize(std::shared_ptr<data::DataTable> const &cfg) {
    _try_invoke_Deserialize<Policies...>(this, cfg);
    DomainBase::Deserialize(cfg);
};
template <template <typename> class... Policies>
template <typename LHS, typename RHS>
void Domain<Policies...>::FillRange(LHS &lhs, RHS &&rhs, std::string const &k) const {
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
