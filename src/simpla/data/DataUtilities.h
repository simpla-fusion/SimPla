//
// Created by salmon on 17-8-20.
//

#ifndef SIMPLA_DATAUTILITIES_H
#define SIMPLA_DATAUTILITIES_H

#include "DataBlock.h"
#include "DataEntity.h"
#include "DataLight.h"
#include "DataNode.h"
#include "DataTraits.h"
namespace simpla {
namespace data {
class KeyValue;
//
// namespace traits {
// inline std::shared_ptr<DataEntity> data_cast(std::shared_ptr<DataEntity> const& u) { return u; }
//
// template <typename U>
// struct DataCast<U> {
//    static std::shared_ptr<DataEntity> cast(U const& u) { return DataLightT<U>::New(u); }
//};
// template <>
// struct DataCast<char const*> {
//    static std::shared_ptr<DataEntity> cast(char const* u) { return DataLightT<std::string>::New((u)); }
//};
// inline std::shared_ptr<DataEntity> data_cast(std::initializer_list<KeyValue> const& u) {
//    auto res = DataTable::New();
//    for (KeyValue const& v : u) { res->Set(v.first, v.second); }
//    return std::dynamic_pointer_cast<DataEntity>(res);
//}
// template <typename... Others>
// std::shared_ptr<DataEntity> data_cast(KeyValue const& first, Others&&... others) {
//    auto res = DataTable::New();
//    res->Set(first, std::forward<Others>(others)...);
//    return res;
//}
//
// template <typename U>
// std::shared_ptr<DataLightArray<U>> data_cast(U const* u, size_type n) {
//    return DataLightArray<U>::New(u, n);
//}
// inline std::shared_ptr<DataLightArray<std::string>> data_cast(std::initializer_list<char const*> const& u) {
//    return DataLightArray<std::string>::New(u);
//}
// template <typename U>
// std::shared_ptr<DataLightArray<U>> data_cast(std::initializer_list<U> const& u,
//                                             ENABLE_IF((traits::is_light_data<U>::value))) {
//    auto p = DataLightArray<U>::New();
//    for (auto const& item : u) { p->Add((item)); }
//    return p;
//}
template <typename U>
std::shared_ptr<DataEntity> make_data_entity(std::shared_ptr<U> const& u,
                                             ENABLE_IF((std::is_base_of<DataEntity, U>::value))) {
    return std::dynamic_pointer_cast<DataEntity>(u);
}
template <typename U>
std::shared_ptr<DataLightT<U>> make_data_entity(U const& u, ENABLE_IF((traits::is_light_data<U>::value))) {
    return DataLightT<U>::New(u);
}

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

template <typename U>
std::shared_ptr<DataLight> make_data_entity(std::initializer_list<U> const& u) {
    std::shared_ptr<DataLight> res = nullptr;
    switch (u.size()) {
        case 0:
            res = DataLight::New();
            break;
        case 1:
            res = make_data_tuple<1>(u);
            break;
        case 2:
            res = make_data_tuple<2>(u);
            break;
        case 3:
            res = make_data_tuple<3>(u);
            break;
        case 4:
            res = make_data_tuple<4>(u);
            break;
        case 5:
            res = make_data_tuple<5>(u);
            break;
        case 6:
            res = make_data_tuple<6>(u);
            break;
        case 7:
            res = make_data_tuple<7>(u);
            break;
        case 8:
            res = make_data_tuple<8>(u);
            break;
        case 9:
            res = make_data_tuple<9>(u);
            break;
        default:
            res = make_data_vector(u);
    }
    return res;
}

template <typename U>
std::shared_ptr<DataLight> make_data_entity(std::initializer_list<std::initializer_list<U>> const& u,
                                            ENABLE_IF((traits::is_light_data<U>::value))) {
    auto p = DataLight::New();
    //    for (auto const& item : u) { p->Add(make_data_entity(item)); }
    return p;
}
template <typename U>
std::shared_ptr<DataLight> make_data_entity(
    std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& u) {
    auto p = DataLight::New();
    //    for (auto const& item : u) { p->Add(make_data_entity(item)); }
    return p;
}

class KeyValue : public std::pair<std::string, std::shared_ptr<DataLight>> {
    typedef std::pair<std::string, std::shared_ptr<DataLight>> base_type;

   public:
    explicit KeyValue(std::string const& k, std::shared_ptr<DataLight> const& p = nullptr) : base_type(k, p) {}
    KeyValue(KeyValue const& other) : base_type(other) {}
    KeyValue(KeyValue&& other) : base_type(other) {}
    ~KeyValue() = default;

    KeyValue& operator=(KeyValue const& other) {
        base_type::operator=(other);
        return *this;
    }

    template <typename U>
    KeyValue& operator=(U const& u) {
        second = make_data_entity(u);
        return *this;
    }
    template <typename U>
    KeyValue& operator=(std::initializer_list<U> const& u) {
        second = make_data_entity(u);
        return *this;
    }
    template <typename U>
    KeyValue& operator=(std::initializer_list<std::initializer_list<U>> const& u) {
        second = make_data_entity(u);
        return *this;
    }
    template <typename U>
    KeyValue& operator=(std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& u) {
        second = make_data_entity(u);
        return *this;
    }
};

inline KeyValue operator"" _(const char* c, std::size_t n) { return KeyValue{std::string(c), make_data_entity(true)}; }

//}
}  // namespace data
}  // namespace simpla
#endif  // SIMPLA_DATAUTILITIES_H
