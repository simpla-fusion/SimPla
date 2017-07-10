//
// Created by salmon on 17-4-5.
//

#ifndef SIMPLA_DOMAINBASE_H
#define SIMPLA_DOMAINBASE_H

#include <simpla/algebra/Array.h>
#include <simpla/algebra/sfc/z_sfc.h>
#include <simpla/data/all.h>
#include <simpla/geometry/Chart.h>
#include <simpla/utilities/Signal.h>
#include <memory>
#include "Attribute.h"

namespace simpla {

namespace geometry {
class GeoObject;
}
namespace engine {
class Patch;
class AttributeGroup;

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

    std::string GetDomainPrefix() const override;

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(std::shared_ptr<data::DataTable> const &t) override;

    void SetBlock(const MeshBlock &);
    const MeshBlock &GetBlock() const;
    id_type GetBlockId() const;

    void Pull(Patch *) override;
    void Push(Patch *) override;

    void SetGeoObject(const std::shared_ptr<geometry::GeoObject> &g);
    const geometry::GeoObject *GetGeoObject() const;

    void SetChart(std::shared_ptr<geometry::Chart> const &g);
    const geometry::Chart *GetChart() const;

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
};  // class DomainBase

template <template <typename> class... Policies>
class Domain : public DomainBase, public Policies<Domain<Policies...>>... {
    typedef Domain<Policies...> host_type;

    SP_OBJECT_HEAD(host_type, DomainBase);

   public:
    typedef DomainBase::attribute_type attribute_type;

    Domain() : Policies<this_type>(this)... {}
    ~Domain() override{};

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

    Domain(const Domain &) = delete;
    Domain(Domain &&) = delete;
    Domain &operator=(Domain const &) = delete;
    Domain &operator=(Domain &&) = delete;

   private:
#define DEFINE_INVOKE_HELPER(_FUN_NAME_)                                                                           \
    template <template <typename> class _T0, typename... Args>                                                     \
    void _invoke_##_FUN_NAME_(Args &&... args) {                                                                   \
        _T0<this_type>::_FUN_NAME_(std::forward<Args>(args)...);                                                   \
    }                                                                                                              \
    template <template <typename> class _T0, template <typename> class _T1, template <typename> class... _TOthers, \
              typename... Args>                                                                                    \
    void _invoke_##_FUN_NAME_(Args &&... args) {                                                                   \
        _invoke_##_FUN_NAME_<_T0>(std::forward<Args>(args)...);                                                    \
        _invoke_##_FUN_NAME_<_T1, _TOthers...>(std::forward<Args>(args)...);                                       \
    }

    DEFINE_INVOKE_HELPER(InitialCondition)
    DEFINE_INVOKE_HELPER(BoundaryCondition)
    DEFINE_INVOKE_HELPER(Advance)
    DEFINE_INVOKE_HELPER(Deserialize)
#undef DEFINE_INVOKE_HELPER
   public:
    void DoInitialCondition(Real time_now) override { _invoke_InitialCondition<Policies...>(time_now); }
    void DoBoundaryCondition(Real time_now, Real dt) override { _invoke_BoundaryCondition<Policies...>(time_now, dt); }
    void DoAdvance(Real time_now, Real dt) override { _invoke_Advance<Policies...>(time_now, dt); }

   private:
    template <template <typename> class _C0>
    void _invoke_Serialize(data::DataTable *cfg) const {
        cfg->Set(_C0<this_type>::Serialize());
    }
    template <template <typename> class _C0, template <typename> class _C1, template <typename> class... _COthers>
    void _invoke_Serialize(data::DataTable *cfg) const {
        _invoke_Serialize<_C0>(cfg);
        _invoke_Serialize<_C1, _COthers...>(cfg);
    }

   public:
    std::shared_ptr<data::DataTable> Serialize() const override {
        auto res = std::make_shared<data::DataTable>();
        _invoke_Serialize<Policies...>(res.get());
        res->Set(DomainBase::Serialize());
        return res;
    };

    void Deserialize(std::shared_ptr<data::DataTable> const &cfg) override {
        _invoke_Deserialize<Policies...>(cfg);
        DomainBase::Deserialize(cfg);
    };

    template <typename TL, typename TR>
    void FillBody(TL &lhs, TR &&rhs) const {
        this->Fill(lhs, std::forward<TR>(rhs));
    };

    template <typename TL, typename TR>
    void FillBoundary(TL &lhs, TR &&rhs) const {
        this->Fill(lhs, std::forward<TR>(rhs));
    };

};  // class Domain

template <template <typename> class... Policies>
bool Domain<Policies...>::is_registered = DomainBase::RegisterCreator<Domain<Policies...>>();

#define DOMAIN_POLICY_HEAD(_NAME_)                                      \
   public:                                                              \
    typedef THost host_type;                                            \
    host_type *m_host_ = nullptr;                                       \
    _NAME_(host_type *h) noexcept : m_host_(h) {}                       \
    virtual ~_NAME_() = default;                                        \
    _NAME_(_NAME_ const &other) = delete;                               \
    _NAME_(_NAME_ &&other) = delete;                                    \
    _NAME_ &operator=(_NAME_ const &other) = delete;                    \
    _NAME_ &operator=(_NAME_ &&other) = delete;                         \
    static std::string RegisterName() { return __STRING(_NAME_); }      \
    void InitialCondition(Real time_now);                               \
    void BoundaryCondition(Real time_now, Real time_dt);                \
    void Advance(Real time_now, Real time_dt);                          \
    virtual std::shared_ptr<simpla::data::DataTable> Serialize() const; \
    virtual void Deserialize(std::shared_ptr<simpla::data::DataTable> const &cfg);

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
