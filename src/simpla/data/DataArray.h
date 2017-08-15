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
   protected:
    explicit DataArray(std::shared_ptr<DataEntity> const& parent = nullptr);

   public:
    ~DataArray() override = default;
    SP_DEFAULT_CONSTRUCT(DataArray)

    static std::shared_ptr<DataArray> New(std::shared_ptr<DataEntity> const& parent = nullptr);

    size_type Count() const override = 0;
    virtual size_type Resize(size_type s) = 0;
    virtual std::shared_ptr<DataEntity> Get(size_type idx) const = 0;
    virtual int Set(size_type idx, std::shared_ptr<DataEntity> const&) = 0;
    virtual int Add(std::shared_ptr<DataEntity> const&) = 0;
    virtual int Delete(size_type idx) = 0;

    virtual int Foreach(std::function<int(std::shared_ptr<DataEntity>)> const& fun) const;
};

struct DataArrayDefault : public DataArray {
    SP_OBJECT_HEAD(DataArrayDefault, DataArray)
   protected:
    explicit DataArrayDefault(std::shared_ptr<DataEntity> const& parent = nullptr);

   public:
    ~DataArrayDefault() override = default;
    SP_DEFAULT_CONSTRUCT(DataArrayDefault);

    static std::shared_ptr<this_type> New(std::shared_ptr<DataEntity> const& parent = nullptr);

    size_type Count() const override;
    size_type Resize(size_type s) override;
    std::shared_ptr<DataEntity> Get(size_type idx) const override;
    int Set(size_type idx, std::shared_ptr<DataEntity> const&) override;
    int Add(std::shared_ptr<DataEntity> const&) override;
    int Delete(size_type idx) override;

   private:
    std::vector<std::shared_ptr<DataEntity>> m_data_;
};
template <typename V, typename Enable = void>
struct DataArrayWrapper {};

template <typename V>
class DataArrayWrapper<V, std::enable_if_t<traits::is_light_data<V>::value>> : public DataArray {
    typedef DataArrayWrapper<V> this_type;
    typedef V value_type;
    std::vector<value_type> m_data_;

   protected:
    explicit DataArrayWrapper(std::shared_ptr<DataEntity> const& parent = nullptr) : DataArray(parent) {}

   public:
    ~DataArrayWrapper() override = default;

    SP_DEFAULT_CONSTRUCT(DataArrayWrapper);

    std::type_info const& value_type_info() const override { return typeid(value_type); };

    std::vector<value_type>& data() { return m_data_; }
    std::vector<value_type> const& data() const { return m_data_; }
    static std::shared_ptr<this_type> New(std::shared_ptr<DataEntity> const& parent = nullptr) {
        return std::shared_ptr<this_type>(new this_type(parent));
    };

    size_type Count() const override { return m_data_.size(); }
    size_type Resize(size_type s = 0) override {
        m_data_.resize(s);
        return s;
    }

    std::shared_ptr<DataEntity> Get(size_type idx) const override { return DataEntity::New(m_data_[idx]); }

    int Set(size_type idx, std::shared_ptr<DataEntity> const& v) override {
        bool success = true;
        if (std::dynamic_pointer_cast<DataEntityWrapper<value_type>>(v) != nullptr) {
            m_data_.at(idx) = std::dynamic_pointer_cast<DataEntityWrapper<value_type>>(v)->value();
        } else {
            success = false;
        }
        return success;
    }
    int Add(std::shared_ptr<DataEntity> const& v) override {
        bool success = true;
        if (std::dynamic_pointer_cast<DataArrayWrapper<value_type>>(v) != nullptr) {
            auto p_array = std::dynamic_pointer_cast<DataArrayWrapper<value_type>>(v);
            for (size_type i = 0, ie = p_array->Count(); i < ie; ++i) { m_data_.push_back(p_array->GetValue(i)); }
        } else if (std::dynamic_pointer_cast<DataEntityWrapper<value_type>>(v) != nullptr) {
            m_data_.push_back(std::dynamic_pointer_cast<DataEntityWrapper<value_type>>(v)->value());
        } else {
            success = false;
        }
        return success;
    }
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
    void Add(value_type const& v) { m_data_.push_back(v); }

    value_type* get() { return &m_data_[0]; }
    value_type const* get() const { return &m_data_[0]; }
};

template <typename U>
std::shared_ptr<DataArrayWrapper<U>> make_data_entity(std::initializer_list<U> const& u,
                                                      ENABLE_IF(traits::is_light_data<U>::value)) {
    auto res = DataArrayWrapper<U>::New();
    for (auto const& item : u) { res->Add(item); }
    return res;
}

inline std::shared_ptr<DataArrayWrapper<std::string>> make_data_entity(std::initializer_list<const char*> const& u) {
    auto res = DataArrayWrapper<std::string>::New();
    for (auto const& item : u) { res->Add(std::string(item)); }
    return res;
}
template <typename U>
std::shared_ptr<DataArray> make_data_entity(std::initializer_list<U> const& u,
                                            ENABLE_IF(!traits::is_light_data<U>::value)) {
    auto res = DataArray::New();
    for (auto const& item : u) { res->Add(make_data_entity(item)); }
    return res;
}

// inline std::shared_ptr<DataArrayWrapper<std::string>> make_data_entity(std::initializer_list<char const*> const& u) {
//    auto res = std::make_shared<DataArrayWrapper<std::string>>();
//    for (auto const item : u) { res->Add(std::string(item)); }
//    return res;
//}

// template <typename U, int N>
// struct DataCastTraits<nTuple<U, N>> {
//    static nTuple<U, N> Get(std::shared_ptr<DataEntity> const& p) {
//        ASSERT(dynamic_cast<DataEntityWrapper<U*> const*>(p.get()) != nullptr);
//        auto a = std::dynamic_pointer_cast<DataEntityWrapper<U*>>(p);
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
// class DataEntityWrapper<simpla::algebra::declare::nTuple_<U, N>> : public DataArrayWithType<U> {
//    typedef simpla::algebra::declare::nTuple_<U, N> tuple_type;
//    SP_OBJECT_HEAD(DataEntityWrapper<tuple_type>, DataArrayWithType<U>);
//    tuple_type m_holder_;
//
//   public:
//    DataEntityWrapper() {}
//    DataEntityWrapper(tuple_type const& other) : m_holder_(other) {}
//    DataEntityWrapper(this_type const& other) : m_holder_(other.m_holder_) {}
//    DataEntityWrapper(this_type&& other) : m_holder_(other.m_holder_) {}
//    virtual ~DataEntityWrapper() {}
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
// class DataEntityWrapper<simpla::algebra::declare::nTuple_<U, N0, N1, N...>> : public DataArray {
//    typedef simpla::algebra::declare::nTuple_<U, N0, N1, N...> tuple_type;
//    SP_OBJECT_HEAD(tuple_type, DataArray);
//    tuple_type m_holder_;
//
//   public:
//    DataEntityWrapper() {}
//    DataEntityWrapper(this_type const& other) : m_holder_(other.m_holder_) {}
//    DataEntityWrapper(this_type&& other) : m_holder_(other.m_holder_) {}
//
//    virtual ~DataEntityWrapper() {}
//    tuple_type& get() { return m_holder_; }
//    tuple_type const& get() const { return m_holder_; }
//    // DataEntity
//    virtual std::shared_ptr<DataEntity> Duplicate() { return std::make_shared<DataEntityWrapper<U*>>(*this); }
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
//    auto p = std::make_shared<DataEntityWrapper<void*>>();
//    p->resize(sizeof...(U));
//    detail::data_entity_to_helper(v, *p, std::integral_constant<int, sizeof...(U)>());
//    return p;
//}

}  // namespace data{
}  // namespace simpla{
#endif  // SIMPLA_DATAARRAY_H
