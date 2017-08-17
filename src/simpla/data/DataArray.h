//
// Created by salmon on 17-3-8.
//

#ifndef SIMPLA_DATAARRAY_H
#define SIMPLA_DATAARRAY_H
#include <functional>
#include "DataEntity.h"
#include "DataTraits.h"

namespace simpla {
namespace data {
struct DataArray : public DataEntity {
    SP_OBJECT_HEAD(DataArray, DataEntity)

    struct pimpl_s;
    pimpl_s* m_pimpl_ = nullptr;

   protected:
    DataArray() = default;

   public:
    ~DataArray() override = default;
    SP_DEFAULT_CONSTRUCT(DataArray)

    static std::shared_ptr<DataArray> New();

    template <typename... Args>
    static std::shared_ptr<DataArray> New(Args&&... args);

    virtual bool isEmpty() const = 0;
    virtual size_type Count() const = 0;
    virtual size_type Resize(size_type s) = 0;
    virtual void Clear() = 0;

    virtual std::shared_ptr<DataEntity> Get(size_type idx) const = 0;
    virtual int Set(size_type idx, std::shared_ptr<DataEntity> const&) = 0;
    virtual int Add(std::shared_ptr<DataEntity> const&) = 0;
    virtual int Delete(size_type idx) = 0;

    virtual int Foreach(std::function<int(std::shared_ptr<DataEntity>)> const& fun) const;
};

template <typename V>
class DataArrayT : public DataArray {
    typedef DataArrayT<V> this_type;
    typedef V value_type;
    std::vector<value_type> m_data_;

   protected:
    DataArrayT() = default;
    template <typename U>
    DataArrayT(std::initializer_list<U> const& u) {
        m_data_.reserve(u.size());
        for (auto const& item : u) { m_data_.push_back(static_cast<value_type>(item)); }
    }
    template <typename U>
    DataArrayT(U const* u, size_type n) : m_data_(n) {
        for (int i = 0; i < n; ++i) { m_data_[i] = u[i]; }
    }

   public:
    ~DataArrayT() override = default;
    SP_DEFAULT_CONSTRUCT(DataArrayT);

    template <typename... Args>
    static std::shared_ptr<this_type> New(Args&&... args) {
        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...));
    };

    std::type_info const& value_type_info() const override { return typeid(value_type); };

    bool isEmpty() const override { return m_data_.empty(); }

    size_type Count() const override { return m_data_.size(); }

    size_type Resize(size_type s) override {
        m_data_.resize(s);
        return s;
    }
    void Clear() override { m_data_.clear(); }

    std::shared_ptr<DataEntity> Get(size_type idx) const override { return DataEntity::New(m_data_[idx]); }

    int Set(size_type idx, std::shared_ptr<DataEntity> const& v) override {
        int count = 0;
        if (auto p = std::dynamic_pointer_cast<DataLight>(v)) {
            m_data_.at(idx) = p->as<value_type>();
            count = 1;
        }
        return count;
    }
    int Add(std::shared_ptr<DataEntity> const& v) override {
        int count = 0;
        if (auto p = std::dynamic_pointer_cast<DataArrayT<value_type>>(v)) {
            count = static_cast<int>(p->Count());
            for (int i = 0; i < count; ++i) { m_data_.push_back(p->GetValue(i)); }
        } else if (auto p = std::dynamic_pointer_cast<DataLight>(v)) {
            m_data_.push_back(p->as<value_type>());
            count = 1;
        }
        return count;
    }
    void Add(value_type const& v) { m_data_.push_back(v); }

    int Delete(size_type idx) override {
        m_data_.erase(m_data_.begin() + idx);
        return 1;
    }

    value_type& GetValue(index_type idx) { return m_data_[idx]; }
    value_type const& GetValue(index_type idx) const { return m_data_[idx]; }
    void SetValue(size_type idx, value_type v) {
        if (Count() < idx) { m_data_.resize(idx); }
        m_data_[idx] = v;
    }

    std::vector<value_type>& data() { return m_data_; }
    std::vector<value_type> const& data() const { return m_data_; }
};
template <>
class DataArrayT<void> : public DataArray {
    typedef DataArrayT<void> this_type;
    std::vector<std::shared_ptr<DataEntity>> m_data_;

   protected:
    DataArrayT() = default;
    template <typename U>
    DataArrayT(std::initializer_list<U> const& u);

   public:
    ~DataArrayT() override = default;
    SP_DEFAULT_CONSTRUCT(DataArrayT);

    template <typename... Args>
    static std::shared_ptr<this_type> New(Args&&... args) {
        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...));
    };

    std::type_info const& value_type_info() const override { return typeid(void); };

    bool isEmpty() const override { return m_data_.empty(); }

    size_type Count() const override { return m_data_.size(); }

    size_type Resize(size_type s) override {
        m_data_.resize(s);
        return s;
    }
    void Clear() override { m_data_.clear(); }

    std::shared_ptr<DataEntity> Get(size_type idx) const override { return m_data_[idx]; }

    int Set(size_type idx, std::shared_ptr<DataEntity> const& v) override {
        bool success = true;
        if (std::dynamic_pointer_cast<DataLight>(v) != nullptr) {
            m_data_.at(idx) = v;
        } else {
            success = false;
        }
        return success;
    }
    int Add(std::shared_ptr<DataEntity> const& v) override {
        m_data_.push_back(v);
        return SP_SUCCESS;
    }
    int Delete(size_type idx) override {
        m_data_.erase(m_data_.begin() + idx);
        return 1;
    }
};
inline std::shared_ptr<DataArray> DataArray::New() { return DataArrayT<void>::New(); }

template <typename U>
DataArrayT<void>::DataArrayT(std::initializer_list<U> const& u) {
    m_data_.reserve(u.size());
    for (auto const& item : u) { m_data_.push_back(DataEntity::New(item)); }
}

template <typename U>
std::shared_ptr<DataArray> make_data_entity(U const* u, size_type n) {
    return DataArrayT<U>::New(u, n);
}
inline std::shared_ptr<DataArrayT<std::string>> make_data_entity(std::initializer_list<char const*> const& u) {
    return DataArrayT<std::string>::New(u);
}
template <typename U>
std::shared_ptr<DataArrayT<U>> make_data_entity(std::initializer_list<U> const& u,
                                                ENABLE_IF((traits::is_light_data<U>::value))) {
    auto p = DataArrayT<U>::New();
    for (auto const& item : u) { p->Add((item)); }
    return p;
}
template <typename U>
std::shared_ptr<DataArrayT<void>> make_data_entity(std::initializer_list<U> const& u,
                                                   ENABLE_IF((!traits::is_light_data<U>::value))) {
    auto p = DataArrayT<void>::New();
    for (auto const& item : u) { p->Add(make_data_entity(item)); }
    return p;
}

template <typename U>
std::shared_ptr<DataArray> make_data_entity(std::initializer_list<std::initializer_list<U>> const& u) {
    return DataArrayT<void>::New(u);
}
template <typename U>
std::shared_ptr<DataArray> make_data_entity(
    std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& u) {
    return DataArrayT<void>::New(u);
}

//
// inline std::shared_ptr<DataArrayT<std::string>> make_data_entity(std::initializer_list<const char*> const& u) {
//    return DataArrayT<std::string>::New(u);
//}
//
// inline std::shared_ptr<DataArrayT<std::string>> make_data_entity(std::initializer_list<char const*> const& u) {
//    auto res = std::make_shared<DataArrayT<std::string>>();
//    for (auto const item : u) { res->Add(std::string(item)); }
//    return res;
//}

// template <typename U, int N>
// struct DataCastTraits<nTuple<U, N>> {
//    static nTuple<U, N> Get(std::shared_ptr<DataEntity> const& p) {
//        ASSERT(dynamic_cast<DataLight<U*> const*>(p.get()) != nullptr);
//        auto a = std::dynamic_pointer_cast<DataLight<U*>>(p);
//        nTuple<U, N> res;
//        for (int i = 0; i < N; ++i) { res[i] = i < a->size() ? DataCastTraits<U>::Get(a->Get(i)) : 0; }
//        return std::move(res);
//    }
//    static nTuple<U, N> Get(std::shared_ptr<DataEntity> const& p, nTuple<U, N> const& default_value) {
//        return (dynamic_cast<DataArray const*>(p.get()) != nullptr) ? Get(p) : default_value;
//    }
//};

//
// template <typename U, int N>
// class DataLight<simpla::algebra::declare::nTuple_<U, N>> : public DataArrayWithType<U> {
//    typedef simpla::algebra::declare::nTuple_<U, N> tuple_type;
//    SP_OBJECT_HEAD(DataLight<tuple_type>, DataArrayWithType<U>);
//    tuple_type m_holder_;
//
//   public:
//    DataLight() {}
//    DataLight(tuple_type const& other) : m_holder_(other) {}
//    DataLight(this_type const& other) : m_holder_(other.m_holder_) {}
//    DataLight(this_type&& other) : m_holder_(other.m_holder_) {}
//    virtual DataLight {}
//    tuple_type& get() { return m_holder_; }
//    tuple_type const& get() const { return m_holder_; }
//    // DataEntity
//    virtual std::shared_ptr<DataEntity> Duplicate() { return std::make_shared<this_type>(*this); }
//
//    // DataArray
//    virtual size_type size() const { return static_cast<size_type>(N); };
//    virtual size_type Foreach(std::function<void(std::shared_ptr<DataEntity>)> const& fun) const {
//        for (size_type s = 0; s < N; ++s) { fun(make_data_entity(m_holder_[s])); }
//        return static_cast<size_type>(N);
//    };
//    virtual void Add(std::shared_ptr<DataEntity> const&) { UNSUPPORTED; };
//    virtual void DeletePatch(size_type idx) { UNSUPPORTED; };
//    // DataArrayWithType
//
//    virtual U GetValue(index_type idx) const { return m_holder_[idx]; }
//    virtual void Deserialize(size_type idx, U const& v) {
//        ASSERT(size() > idx);
//        m_holder_[idx] = v;
//    }
//    virtual void Add(U const&) { UNSUPPORTED; };
//
//    tuple_type value() const { return m_holder_; }
//};
//
// template <typename U, int N0, int N1, int... N>
// class DataLight<simpla::algebra::declare::nTuple_<U, N0, N1, N...>> : public DataArray {
//    typedef simpla::algebra::declare::nTuple_<U, N0, N1, N...> tuple_type;
//    SP_OBJECT_HEAD(tuple_type, DataArray);
//    tuple_type m_holder_;
//
//   public:
//    DataLight() {}
//    DataLight(this_type const& other) : m_holder_(other.m_holder_) {}
//    DataLight(this_type&& other) : m_holder_(other.m_holder_) {}
//
//    virtual DataLight {}
//    tuple_type& get() { return m_holder_; }
//    tuple_type const& get() const { return m_holder_; }
//    // DataEntity
//    virtual std::shared_ptr<DataEntity> Duplicate() { return std::make_shared<DataLight<U*>>(*this); }
//    // DataArray
//    virtual size_type size() const {
//        UNIMPLEMENTED;
//        return N0 * N1;
//    };
//    virtual size_type Foreach(std::function<void(std::shared_ptr<DataEntity>)> const& fun) const {
//        UNIMPLEMENTED;
//        return size();
//    };
//    virtual void Add(std::shared_ptr<DataEntity> const&) { UNSUPPORTED; };
//    virtual void DeletePatch(size_type idx) { UNSUPPORTED; };
//    // DataArrayWithType
//    virtual U GetValue(index_type idx) const { return m_holder_[idx]; }
//
//    virtual void Deserialize(size_type idx, U const& v) {
//        ASSERT(size() > idx);
//        m_holder_[idx] = v;
//    }
//    virtual void Add(U const&) { UNSUPPORTED; };
//};

//
// namespace detail {
// template <typename V>
// void data_entity_from_helper0(DataEntity const& v, V& u) {
//    u = data_cast<V>(v);
//}
// template <typename... U>
// void data_entity_from_helper(DataArray const&, std::tuple<U...>& v, std::integral_constant<int, 0>){};
//
// template <int N, typename... U>
// void data_entity_from_helper(DataArray const& a, std::tuple<U...>& v, std::integral_constant<int, N>) {
//    data_entity_from_helper0(*a.Serialize(N - 1), std::get<N - 1>(v));
//    data_entity_from_helper(a, v, std::integral_constant<int, N - 1>());
//};
//
// template <typename V>
// void data_entity_to_helper0(V const& src, DataArray& dest, size_type N) {
//    dest.Deserialize(N - 1, data_entity_traits<V>::to(src));
//}
// template <typename... U>
// void data_entity_to_helper(std::tuple<U...> const& src, DataArray& dest, std::integral_constant<int, 0>){};
//
// template <int N, typename... U>
// void data_entity_to_helper(std::tuple<U...> const& src, DataArray& dest, std::integral_constant<int, N>) {
//    data_entity_to_helper0(std::get<N - 1>(src), dest, N);
//    data_entity_to_helper(src, dest, std::integral_constant<int, N - 1>());
//};
//}
// template <typename... U>
// std::shared_ptr<DataEntity> make_data_entity(std::tuple<U...> const& v) {
//    auto p = std::make_shared<DataLight<void*>>();
//    p->resize(sizeof...(U));
//    detail::data_entity_to_helper(v, *p, std::integral_constant<int, sizeof...(U)>());
//    return p;
//}

}  // namespace data{
}  // namespace simpla{
#endif  // SIMPLA_DATAARRAY_H
