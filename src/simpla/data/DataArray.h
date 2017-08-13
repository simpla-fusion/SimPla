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

   public:
    DataArray() = default;
    ~DataArray() override = default;
    DataArray(DataArray const& other) = delete;
    DataArray(DataArray&& other) = delete;
    DataArray& operator=(DataArray const& other) = delete;
    DataArray& operator=(DataArray&& other) = delete;

    std::shared_ptr<DataEntity> Duplicate() const override = 0;

    std::ostream& Serialize(std::ostream& os, int indent) const override {
        if (size() == 0) { return os; };

        os << "[";
        Get(0)->Serialize(os, indent + 1);
        for (size_type i = 0, ie = size(); i < ie; ++i) {
            os << ",";
            Get(i)->Serialize(os, indent + 1);
        }
        os << "]";
        return os;
    }
    /**   DataArray */
    virtual size_type size() const = 0;
    virtual void resize(size_type s = 0) = 0;
    virtual std::shared_ptr<DataEntity> Get(size_type idx) = 0;
    virtual std::shared_ptr<DataEntity> Get(size_type idx) const = 0;
    virtual bool Set(size_type idx, std::shared_ptr<DataEntity> const&) = 0;
    virtual bool Add(std::shared_ptr<DataEntity> const&) = 0;
    virtual bool Delete(size_type idx) = 0;

    virtual size_type Foreach(std::function<void(std::shared_ptr<DataEntity>)> const& fun) const { return 0; }
};
template <typename V = void, typename Enable = void>
struct DataArrayWrapper {};

template <>
struct DataArrayWrapper<void, void> : public DataArray {
    typedef DataArrayWrapper<void, void> this_type;

   public:
    DataArrayWrapper() = default;
    ~DataArrayWrapper() override = default;
    DataArrayWrapper(DataArrayWrapper const& other) : m_data_(other.m_data_){};
    DataArrayWrapper(DataArrayWrapper&& other) = delete;
    DataArrayWrapper& operator=(DataArrayWrapper const& other) = delete;
    DataArrayWrapper& operator=(DataArrayWrapper&& other) = delete;

    template <typename U>
    DataArrayWrapper(std::initializer_list<U> const& v) {
        for (auto const& item : v) { m_data_.push_back(make_data_entity(item)); }
    };
    std::shared_ptr<DataEntity> Duplicate() const override {
        return std::dynamic_pointer_cast<DataEntity>(std::make_shared<this_type>(*this));
    }

    auto& get() { return m_data_; }
    auto const& get() const { return m_data_; }

    void resize(size_type s) override { m_data_.resize(s); }
    size_type size() const override { return m_data_.size(); }
    std::shared_ptr<DataEntity> Get(size_type idx) override { return m_data_[idx]; }
    std::shared_ptr<DataEntity> Get(size_type idx) const override { return m_data_.at(idx); }
    bool Set(size_type idx, std::shared_ptr<DataEntity> const& v) override {
        bool success = idx < m_data_.size();
        if (success) { m_data_.at(idx) = v; }
        return success;
    }
    bool Add(std::shared_ptr<DataEntity> const& v) override {
        if (std::dynamic_pointer_cast<DataArray>(v) != nullptr) {
            auto p_array = std::dynamic_pointer_cast<DataArray>(v);
            for (size_type i = 0, ie = p_array->size(); i < ie; ++i) { m_data_.push_back(p_array->Get(i)); }
        } else {
            m_data_.push_back(v);
        }
        return true;
    }
    bool Delete(size_type idx) override {
        m_data_.erase(m_data_.begin() + idx);
        return true;
    }

   private:
    std::vector<std::shared_ptr<DataEntity>> m_data_;
};

template <typename U>
class DataArrayWrapper<U> : public DataArray {
    SP_OBJECT_HEAD(DataArrayWrapper<U>, DataArray);
    typedef U value_type;
    std::vector<U> m_data_;

   public:
    DataArrayWrapper() = default;
    ~DataArrayWrapper() override = default;

    DataArrayWrapper(std::initializer_list<U> const& l) : m_data_(l){};

    template <typename... Args>
    explicit DataArrayWrapper(Args&&... args) : m_data_(std::forward<Args>(args)...) {}

    DataArrayWrapper(DataArrayWrapper const& other) : m_data_(other.m_data_){};
    DataArrayWrapper(DataArrayWrapper&& other) = delete;
    DataArrayWrapper& operator=(DataArrayWrapper const& other) = delete;
    DataArrayWrapper& operator=(DataArrayWrapper&& other) = delete;
    std::shared_ptr<DataEntity> Duplicate() const override {
        return std::dynamic_pointer_cast<DataEntity>(std::make_shared<this_type>(*this));
    }

    U* get() { return &m_data_[0]; }
    U const* get() const { return &m_data_[0]; }

    // DataEntity

    // DataArray
    void resize(size_type s) override { m_data_.resize(s); }
    size_type size() const override { return m_data_.size(); };
    std::shared_ptr<DataEntity> Get(size_type idx) override {
        return std::make_shared<DataEntityWrapper<value_type>>(&m_data_[idx]);
    }
    std::shared_ptr<DataEntity> Get(size_type idx) const override {
        return std::make_shared<DataEntityWrapper<value_type>>(&m_data_[idx]);
    }
    bool Set(size_type idx, std::shared_ptr<DataEntity> const& v) override {
        bool success = true;
        if (std::dynamic_pointer_cast<DataEntityWrapper<value_type>>(v) != nullptr) {
            m_data_.at(idx) = std::dynamic_pointer_cast<DataEntityWrapper<value_type>>(v)->value();
        } else {
            success = false;
        }
        return success;
    }
    bool Add(std::shared_ptr<DataEntity> const& v) override {
        bool success = true;
        if (std::dynamic_pointer_cast<DataArrayWrapper<value_type>>(v) != nullptr) {
            auto p_array = std::dynamic_pointer_cast<DataArrayWrapper<value_type>>(v);
            for (size_type i = 0, ie = p_array->size(); i < ie; ++i) { m_data_.push_back(p_array->GetValue(i)); }
        } else if (std::dynamic_pointer_cast<DataEntityWrapper<value_type>>(v) != nullptr) {
            m_data_.push_back(std::dynamic_pointer_cast<DataEntityWrapper<value_type>>(v)->value());
        } else {
            success = false;
        }
        return success;
    }
    bool Delete(size_type idx) override {
        m_data_.erase(m_data_.begin() + idx);
        return true;
    }

    U& GetValue(index_type idx) { return m_data_[idx]; }
    U const& GetValue(index_type idx) const { return m_data_[idx]; }
    void SetValue(size_type idx, U v) {
        if (size() < idx) { m_data_.resize(idx); }
        m_data_[idx] = v;
    }
    void Add(U const& v) { m_data_.push_back(v); }
};

template <typename U>
std::shared_ptr<DataArrayWrapper<U>> make_data_entity(std::initializer_list<U> const& u,
                                                      ENABLE_IF(traits::is_light_data<U>::value)) {
    return std::make_shared<DataArrayWrapper<U>>(u);
}
template <typename U>
std::shared_ptr<DataArrayWrapper<>> make_data_entity(std::initializer_list<U> const& u,
                                                      ENABLE_IF(!traits::is_light_data<U>::value)) {
    auto res = std::make_shared<DataArrayWrapper<>>();
    for (auto const& item : u) { res->Add(make_data_entity(item)); }
    return res;
}

//inline std::shared_ptr<DataArrayWrapper<std::string>> make_data_entity(std::initializer_list<char const*> const& u) {
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
