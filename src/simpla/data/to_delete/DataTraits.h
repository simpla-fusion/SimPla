//
// Created by salmon on 17-3-2.
//

#ifndef SIMPLA_DATATRAITS_H
#define SIMPLA_DATATRAITS_H

#include "DataEntity.h"
#include "KeyValue.h"
namespace simpla {
namespace data {

template <typename U>
U& data_cast(DataEntity& v) {
    if (!v.isLight()) { RUNTIME_ERROR << "illegal type convert" << std::endl; }
    return v.as<LightData>()->GetValue<U>();
}
template <typename U>
U const& data_cast(DataEntity const& v) {
    if (!v.isLight()) { RUNTIME_ERROR << "illegal type convert" << std::endl; }
    return v.as<LightData>()->GetValue<U>();
}

template <typename T>
bool Check(DataEntity const& v, T const& u) {
    if (!v.isLight()) { RUNTIME_ERROR << "illegal type convert" << std::endl; }
    return v.template as<LightData>()->equal(u);
};

template <typename U>
struct DataTraits {
    static U& from(DataEntity& d) { return d.as<LightData>()->as<U>(); };
    static U const& from(DataEntity const& d) { return d.as<LightData>()->as<U>(); };
    static std::shared_ptr<DataEntity> to(U const& u) { return std::make_shared<LightData>(u); }
    static std::shared_ptr<DataEntity> to(U& u) { return std::make_shared<LightData>(u); }
};

// std::string GetValue(std::string const& url, char const* u) const {
//    auto p = find(url);
//    return p == nullptr ? std::string(u) : p->template GetValue<std::string>();
//}
//    template<typename U> U const &GetValue(std::string const &url, U const &u)
//    {
//        auto *p = find(url);
//        if (p != nullptr) { return p->as<U>(); } else { return Set(url, u)->as<U>();}
//    }

}  // namespace data{
}  // namespace simpla
#endif  // SIMPLA_DATATRAITS_H
