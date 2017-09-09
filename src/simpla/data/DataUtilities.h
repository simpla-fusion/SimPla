//
// Created by salmon on 17-8-20.
//

#ifndef SIMPLA_DATAUTILITIES_H
#define SIMPLA_DATAUTILITIES_H

#include "DataBlock.h"
#include "DataEntity.h"
#include "DataLight.h"
#include "DataTraits.h"
namespace simpla {
template <typename T, int...>
struct nTuple;
namespace data {
class KeyValue;
template <typename U>
U data_cast(std::shared_ptr<DataEntity> const& ptr) {
    U res;
    size_type count = 0;
    if (auto p = std::dynamic_pointer_cast<DataLightT<U>>(ptr)) {
        count = p->CopyOut(res);
    } else if (auto p = std::dynamic_pointer_cast<DataLightT<traits::value_type_t<U>>>(ptr)) {
        count = p->CopyOut(res);
    } else if (auto p = std::dynamic_pointer_cast<DataBlockT<traits::value_type_t<U>>>(ptr)) {
        count = p->CopyOut(res);
    }
    if (count == 0) { BAD_CAST; }
    return res;
}
template <typename, typename>
struct DataLightT;

template <typename U>
std::shared_ptr<DataEntity> make_data_entity(std::shared_ptr<U> const& u,
                                             ENABLE_IF((std::is_base_of<DataEntity, U>::value))) {
    return std::dynamic_pointer_cast<DataEntity>(u);
}
template <typename U>
std::shared_ptr<DataLightT<U>> make_data_entity(U const& u, ENABLE_IF((traits::is_light_data<U>::value))) {
    return DataLightT<U>::New(u);
}
template <typename U, int... N>
auto make_data_entity(std::tuple<nTuple<U, N...>, nTuple<U, N...>> const& u,
                      ENABLE_IF((traits::is_light_data<U>::value))) {
    nTuple<U, 2, N...> res;
    res[0] = std::get<0>(u);
    res[1] = std::get<1>(u);

    return make_data_entity(res);
}
template <typename U>
std::shared_ptr<DataLight> make_data_entity(std::initializer_list<U> const& u) {
    size_type s = u.size();
    auto res = DataLightT<U*>::New(1, &s);
    size_type i = 0;
    for (auto const& v : u) {
        res->pointer()[i] = v;
        ++i;
    }
    return res;
}

template <typename U>
std::shared_ptr<DataEntity> make_data_entity(std::initializer_list<std::initializer_list<U>> const& u) {
    return nullptr;
};
template <typename U>
std::shared_ptr<DataEntity> make_data_entity(
    std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& u) {
    return nullptr;
};
inline std::shared_ptr<DataLightT<std::string>> make_data_entity(char const* c) {
    return DataLightT<std::string>::New(std::string(c));
}
template <int N, typename U>
std::shared_ptr<DataLightT<nTuple<U, N>>> make_data_tuple(std::initializer_list<U> const& u) {
    auto res = DataLightT<nTuple<U, N>>::New();
    int count = 0;
    for (auto const& v : u) {
        res->value()[count] = v;
        ++count;
    }
    return res;
};
template <typename U>
std::shared_ptr<DataLightT<std::vector<U>>> make_data_vector(std::initializer_list<U> const& u) {
    auto res = DataLightT<std::vector<U>>::New();
    for (auto const& v : u) { res->value().push_back(v); }
    return res;
};

// KeyValue const& make_data(KeyValue const& u) { return u; }
// std::initializer_list<KeyValue> const& make_data(std::initializer_list<KeyValue> const& u) { return u; }
// std::initializer_list<std::initializer_list<KeyValue>> const& make_data(
//    std::initializer_list<std::initializer_list<KeyValue>> const& u) {
//    return u;
//}
// std::initializer_list<std::initializer_list<std::initializer_list<KeyValue>>> const& make_data(
//    std::initializer_list<std::initializer_list<std::initializer_list<KeyValue>>> const& u) {
//    return u;
//}
//    int Set(KeyValue const& kv) { return GetNode(kv.first, RECURSIVE | NEW_IF_NOT_EXIST)->SetEntity(kv.m_node_); }
//
//    template <typename... Others>
//    int SetEntity(KeyValue const& kv, Others&&... others) {
//        return Set(kv) + SetEntity(std::forward<Others>(others)...);
//    }
//    int SetEntity(std::initializer_list<KeyValue> const& u) {
//        int count = 0;
//        for (auto const& item : u) { count += (SetEntity(item)); }
//        return count;
//    }
//    int SetEntity(std::initializer_list<std::initializer_list<KeyValue>> const& u) {
//        int count = 0;
//        for (auto const& item : u) { count += (SetEntity(item)); }
//        return count;
//    }
//    int SetEntity(std::initializer_list<std::initializer_list<std::initializer_list<KeyValue>>> const& u) {
//        int count = 0;
//        for (auto const& item : u) { count += (SetEntity(item)); }
//        return count;
//    }
}  // namespace data
}  // namespace simpla
#endif  // SIMPLA_DATAUTILITIES_H
