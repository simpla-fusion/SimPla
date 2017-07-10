//
// Created by salmon on 17-7-10.
//

#ifndef SIMPLA_DOMAIN_H
#define SIMPLA_DOMAIN_H

#include <simpla/utilities/ObjectHead.h>
#include "DomainBase.h"
namespace simpla {
namespace engine {
template <template <typename> class... Policies>
class Domain : public DomainBase, public Policies<Domain<Policies...>>... {
    typedef Domain<Policies...> host_type;

    SP_OBJECT_HEAD(host_type, DomainBase);

   public:
    using DomainBase::attribute_type;

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
        cfg->Link(_C0<this_type>::Serialize());
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
        return res;
    };

    void Deserialize(std::shared_ptr<data::DataTable> const &cfg) override { _invoke_Deserialize<Policies...>(cfg); };

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

}  // namespace engine
}  // namespace simpla
#endif  // SIMPLA_DOMAIN_H
