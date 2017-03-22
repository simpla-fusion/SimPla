//
// Created by salmon on 17-3-8.
//

#ifndef SIMPLA_DATAARRAY_H
#define SIMPLA_DATAARRAY_H
#include "DataEntity.h"
#include "DataTraits.h"
namespace simpla {
namespace data {
template <typename U, typename Enable = void>
class DataArrayWrapper {};
struct DataArray : public DataEntity {
    SP_OBJECT_HEAD(DataArray, DataEntity)
    size_type m_dimensions_[MAX_NDIMS_OF_ARRAY] = {1, 1, 1, 1, 1, 1, 1, 1};

   public:
    DataArray();
    virtual ~DataArray();
    virtual std::ostream& Print(std::ostream& os, int indent = 0) const;
    virtual bool isArray() const { return true; }
    /** as Array */
    virtual size_type rank() const { return 1; };
    virtual size_type const* dimensions() const { return m_dimensions_; };
    virtual size_type size() const {
        UNIMPLEMENTED;
        return 0;
    };
    virtual std::shared_ptr<DataEntity> Get(size_type idx) const {
        UNIMPLEMENTED;
        return std::make_shared<DataEntity>();
    }
    virtual void Set(size_type idx, std::shared_ptr<DataEntity> const&) { UNIMPLEMENTED; }
    virtual void Add(std::shared_ptr<DataEntity> const&) { UNIMPLEMENTED; }
    virtual void Delete(size_type idx) { UNIMPLEMENTED; }

    virtual size_type Foreach(std::function<void(std::shared_ptr<DataEntity>)> const&) const {
        UNIMPLEMENTED;
        return 0;
    };
};
template <>
struct DataArrayWrapper<void> : public DataArray {
    SP_OBJECT_HEAD(DataArrayWrapper<void>, DataArray)
   public:
    DataArrayWrapper() {}

    template <typename U>
    DataArrayWrapper(std::initializer_list<U> const& v) {
        for (auto const& item : v) { m_data_.push_back(make_data_entity(v)); }
    };

    virtual ~DataArrayWrapper(){};
    virtual void resize(size_type s) { m_data_.resize(s); }
    virtual size_type size() const { return m_data_.size(); }
    virtual std::shared_ptr<DataEntity> Get(size_type idx) const { return m_data_[idx]; }
    virtual void Set(size_type idx, std::shared_ptr<DataEntity> const& v) { m_data_[idx] = v; }
    virtual void Add(std::shared_ptr<DataEntity> const& v) { m_data_.push_back(v); }
    virtual void Delete(size_type idx) { m_data_.erase(m_data_.begin() + idx); }
    virtual size_type Foreach(std::function<void(std::shared_ptr<DataEntity>)> const& fun) const {
        for (auto const& item : m_data_) { fun(item); }
        return m_data_.size();
    };

   private:
    std::vector<std::shared_ptr<DataEntity>> m_data_;
};

template <typename U>
class DataArrayWrapper<U, std::enable_if_t<traits::is_light_data<U>::value>> : public DataArray {
    SP_OBJECT_HEAD(DataArrayWrapper<U>, DataArray);
    std::vector<U> m_data_;

   public:
    DataArrayWrapper() {}
    DataArrayWrapper(this_type const& other) : m_data_(other.m_data_) {}
    DataArrayWrapper(this_type&& other) : m_data_(other.m_data_) {}

    virtual ~DataArrayWrapper() {}
    virtual std::type_info const& value_type_info() const { return typeid(U); };
    virtual std::shared_ptr<DataEntity> Duplicate() { return std::make_shared<DataArrayWrapper<U>>(*this); }
    std::vector<U>& data() { return m_data_; }
    std::vector<U> const& data() const { return m_data_; }
    virtual size_type size() const { return m_data_.size(); };

    virtual U const& GetValue(size_type idx) const { return m_data_[idx]; }
    virtual U const& operator[](size_type idx) const { return m_data_[idx]; }
    virtual U& GetValue(size_type idx) { return m_data_[idx]; }
    virtual U& operator[](size_type idx) { return m_data_[idx]; }

    virtual std::shared_ptr<DataEntity> Get(size_type idx) const { return make_data_entity(m_data_[idx]); }

    virtual void Set(size_type idx, U const& v) {
        if (size() > idx) { m_data_[idx] = v; }
    }
    virtual void Set(size_type idx, std::shared_ptr<DataEntity> const& v) { Set(idx, data_cast<U>(*v)); }

    virtual void Add(U const& v) { m_data_.push_back(v); }
    virtual void Add(std::shared_ptr<DataEntity> const& v) { m_data_.push_back(data_cast<U>(*v)); }
    virtual void Delete(size_type idx) { m_data_.erase(m_data_.begin() + idx); }

    virtual size_type Foreach(std::function<void(std::shared_ptr<DataEntity>)> const& fun) const {
        for (auto const& item : m_data_) { fun(make_data_entity(item)); }
        return m_data_.size();
    };
};

// template <typename U>
// std::shared_ptr<DataEntity> make_data_entity(std::initializer_list<U> const& u,
//                                             ENABLE_IF(traits::is_light_data<U>::value)) {
//    auto res = std::make_shared<DataArrayWrapper<U>>();
//    for (U const& v : u) { res->Add(v); }
//    return std::dynamic_pointer_cast<DataEntity>(res);
//}

template <typename U>
struct data_entity_traits<std::initializer_list<U>, std::enable_if_t<traits::is_light_data<U>::value>> {
    static std::shared_ptr<DataEntity> to(std::initializer_list<U> const& l) {
        auto res = std::make_shared<DataArrayWrapper<U>>();
        for (auto const& v : l) { res->Add(v); }
        return std::dynamic_pointer_cast<DataEntity>(res);
    };
};
template <typename U>
std::shared_ptr<DataEntity> make_data_entity(std::initializer_list<U> const& u) {
    return data_entity_traits<std::initializer_list<U>>::to(u);
}

template <typename U>
struct data_entity_traits<std::initializer_list<U>, std::enable_if_t<!traits::is_light_data<U>::value>> {
    static std::shared_ptr<DataEntity> to(std::initializer_list<U> const& l) {
        auto res = std::make_shared<DataArrayWrapper<void>>();
        for (auto const& v : l) { res->Add(make_data_entity(v)); }
        return std::dynamic_pointer_cast<DataEntity>(res);
    };
};
template <>
struct data_entity_traits<std::initializer_list<char const*>> {
    static std::vector<std::string> from(DataEntity const& a_entity) {
        std::vector<std::string> res;
        auto const& l = dynamic_cast<DataArrayWrapper<std::string> const&>(a_entity);
        for (size_type i = 0, ie = l.size(); i < ie; ++i) { res.push_back(l[i]); }
        return res;
    };
    static std::shared_ptr<DataEntity> to(std::initializer_list<const char*> const& l) {
        auto res = std::make_shared<DataArrayWrapper<std::string>>();
        for (char const* v : l) { res->Add(v); }
        return std::dynamic_pointer_cast<DataEntity>(res);
    };
};

template <typename U, int N>
struct data_entity_traits<nTuple<U, N>, std::enable_if_t<traits::is_light_data<U>::value>> {
    static nTuple<U, N> from(DataEntity const& v) {
        nTuple<U, N> res;
        for (int i = 0; i < N; ++i) { res[i] = data_cast<U>(*v.cast_as<DataArray>().Get(i)); };
        return std::move(res);
    };

    static std::shared_ptr<DataEntity> to(nTuple<U, N> const& u) {
        auto res = std::make_shared<DataArrayWrapper<U>>();
        for (int i = 0; i < N; ++i) { res->Add(u[i]); }
        return std::dynamic_pointer_cast<DataEntity>(res);
    }
};
template <typename U, int N, int M>
struct data_entity_traits<nTuple<U, N, M>, std::enable_if_t<traits::is_light_data<U>::value>> {
    static nTuple<U, N, M> from(DataEntity const& v) {
        nTuple<U, N, M> res;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < M; ++j) {
                res[i][j] = data_cast<U>(*v.cast_as<DataArray>().Get(i)->cast_as<DataArray>().Get(j));
            };
        return std::move(res);
    };

    static std::shared_ptr<DataEntity> to(nTuple<U, N, M> const& v) {
        auto res = std::make_shared<DataArrayWrapper<void>>();
        for (int i = 0; i < N; ++i) { res->Add(make_data_entity(v[i])); }
        return std::dynamic_pointer_cast<DataEntity>(res);
    };
};

template <typename U, int... N>
std::shared_ptr<DataEntity> make_data_entity(std::initializer_list<nTuple<U, N...>> const& u) {
    auto res = std::make_shared<DataArrayWrapper<nTuple<U, N...>>>();
    for (nTuple<U, N...> const& v : u) { res->Add(v); }
    return std::dynamic_pointer_cast<DataEntity>(res);
}

namespace detail {
template <typename V>
void data_entity_from_helper0(DataEntity const& v, V& u) {
    u = data_cast<V>(v);
}
template <typename... U>
void data_entity_from_helper(DataArray const&, std::tuple<U...>& v, std::integral_constant<int, 0>){};

template <int N, typename... U>
void data_entity_from_helper(DataArray const& a, std::tuple<U...>& v, std::integral_constant<int, N>) {
    data_entity_from_helper0(*a.Get(N - 1), std::get<N - 1>(v));
    data_entity_from_helper(a, v, std::integral_constant<int, N - 1>());
};

template <typename V>
void data_entity_to_helper0(V const& src, DataArray& dest, size_type N) {
    dest.Set(N - 1, data_entity_traits<V>::to(src));
}
template <typename... U>
void data_entity_to_helper(std::tuple<U...> const& src, DataArray& dest, std::integral_constant<int, 0>){};

template <int N, typename... U>
void data_entity_to_helper(std::tuple<U...> const& src, DataArray& dest, std::integral_constant<int, N>) {
    data_entity_to_helper0(std::get<N - 1>(src), dest, N);
    data_entity_to_helper(src, dest, std::integral_constant<int, N - 1>());
};
}
template <typename... U>
struct data_entity_traits<std::tuple<U...>> {
    static std::tuple<U...> from(DataEntity const& v) {
        std::tuple<U...> res;
        detail::data_entity_from_helper(v.cast_as<DataArray>(), res, std::integral_constant<int, sizeof...(U)>());
        return std::move(res);
    };

    static std::shared_ptr<DataEntity> to(std::tuple<U...> const& v) {
        auto p = std::make_shared<DataArrayWrapper<void>>();
        p->resize(sizeof...(U));
        detail::data_entity_to_helper(v, *p, std::integral_constant<int, sizeof...(U)>());
        return p;
    }
};
}  // namespace data{
}  // namespace simpla{
#endif  // SIMPLA_DATAARRAY_H
