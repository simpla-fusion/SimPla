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
    Domain() : Policies<this_type>(this)... {}
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

    Domain(const Domain &) = delete;
    Domain(Domain &&) = delete;
    Domain &operator=(Domain const &) = delete;
    Domain &operator=(Domain &&) = delete;

    void DoInitialCondition(Real time_now) override {}
    void DoBoundaryCondition(Real time_now, Real dt) override {}
    void DoAdvance(Real time_now, Real dt) override {}

};  // class Domain

}  // namespace engine
}  // namespace simpla
#endif  // SIMPLA_DOMAIN_H
