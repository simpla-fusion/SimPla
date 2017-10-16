//
// Created by salmon on 17-10-16.
//

#ifndef SIMPLA_EBDOMAIN_H
#define SIMPLA_EBDOMAIN_H

#include "Domain.h"
namespace simpla {
namespace engine {

template <typename TChart, template <typename> class... Policies>
struct EBDomain : public Domain<TChart, Policies...> {
    typedef Domain<TChart, Policies...> base_domain;
    SP_OBJECT_HEAD(EBDomain, base_domain)
   protected:
   private:
   public:
    template <template <typename> class U>
    std::shared_ptr<U<this_type>> AddEmbeddedDomain(std::string const &k,
                                                    std::shared_ptr<geometry::GeoObject> const &g) {
        auto res = U<this_type>::New();
        res->SetBoundary(g);
        DomainBase::AddEmbeddedDomain(k, res);
        return res;
    };
};

template <typename TChart, template <typename> class... Policies>
EBDomain<TChart, Policies...>::EBDomain() {}

template <typename TChart, template <typename> class... Policies>
EBDomain<TChart, Policies...>::~EBDomain() {}

template <typename TChart, template <typename> class... Policies>
void EBDomain<TChart, Policies...>::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg){};

template <typename TChart, template <typename> class... Policies>
std::shared_ptr<simpla::data::DataNode> EBDomain<TChart, Policies...>::Serialize() const {
    return base_type::Serialize();
};
}  // namespace engine
}  // namespace simpla
#endif  // SIMPLA_EBDOMAIN_H
