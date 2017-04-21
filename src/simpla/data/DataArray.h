//
// Created by salmon on 17-3-8.
//

#ifndef SIMPLA_DATAARRAY_H
#define SIMPLA_DATAARRAY_H
#include "DataEntity.h"
#include "DataTraits.h"
namespace simpla {
namespace data {

template <typename U>
std::shared_ptr<DataEntity> make_data_entity(std::initializer_list<U> const& u);
template <typename U>
std::shared_ptr<DataEntity> make_data_entity(std::initializer_list<std::initializer_list<U>> const& u);

inline std::shared_ptr<DataEntity> make_data_entity(std::initializer_list<char const*> const& u);

struct DataArray : public DataEntity {
    SP_OBJECT_HEAD(DataArray, DataEntity)

   public:
    DataArray() = default;
    ~DataArray() override = default;
    SP_DEFAULT_CONSTRUCT(DataArray)

    std::ostream& Serialize(std::ostream& os, int indent = 0) const override;
    bool isArray() const override { return true; }
    /**   DataArray */
    virtual size_type size() const { return 0; };
    virtual std::shared_ptr<DataEntity> Get(index_type idx) const { return std::make_shared<DataEntity>(); }
    virtual void Set(size_type idx, std::shared_ptr<DataEntity> const&) { UNIMPLEMENTED; }
    virtual void Add(std::shared_ptr<DataEntity> const&) { UNIMPLEMENTED; }
    virtual void Delete(size_type idx) { UNIMPLEMENTED; }

    virtual size_type Foreach(std::function<void(std::shared_ptr<DataEntity>)> const&) const { return 0; };
};

template <>
struct DataEntityWrapper<void*> : public DataArray {
    SP_OBJECT_HEAD(DataEntityWrapper<void>, DataArray)
   public:
    DataEntityWrapper() = default;
    ~DataEntityWrapper() override = default;
    SP_DEFAULT_CONSTRUCT(DataEntityWrapper)

    template <typename U>
    DataEntityWrapper(std::initializer_list<U> const& v) {
        for (auto const& item : v) { m_data_.push_back(make_data_entity(item)); }
    };

    std::ostream& Serialize(std::ostream& os, int indent) const override {
        if (m_data_.size() == 0) { return os; };
        auto it = m_data_.begin();
        os << "[" << **it;
        for (++it; it != m_data_.end(); ++it) { os << "," << **it; }
        os << "]";
        return os;
    }

    std::vector<std::shared_ptr<DataEntity>>& get() { return m_data_; }
    std::vector<std::shared_ptr<DataEntity>> const& get() const { return m_data_; }

    void resize(size_type s) { m_data_.resize(s); }

    size_type size() const override { return m_data_.size(); }
    std::shared_ptr<DataEntity> Get(index_type idx) const override { return m_data_[idx]; }
    void Set(size_type idx, std::shared_ptr<DataEntity> const& v) override { m_data_[idx] = v; }
    void Add(std::shared_ptr<DataEntity> const& v) override { m_data_.push_back(v); }
    void Delete(size_type idx) override { m_data_.erase(m_data_.begin() + idx); }

    size_type Foreach(std::function<void(std::shared_ptr<DataEntity>)> const& fun) const override {
        for (auto const& item : m_data_) { fun(item); }
        return m_data_.size();
    };

   private:
    std::vector<std::shared_ptr<DataEntity>> m_data_;
};
template <typename U>
class DataArrayWithType : public DataArray {
    typedef U value_type;

    SP_OBJECT_HEAD(DataArrayWithType<U>, DataArray);

   public:
    DataArrayWithType() = default;
    ~DataArrayWithType() override = default;

    SP_DEFAULT_CONSTRUCT(DataArrayWithType)

    // DataEntity
    std::type_info const& value_type_info() const override { return typeid(value_type); }
    bool isLight() const override { return traits::is_light_data<value_type>::value; }

    // DataArray
    std::shared_ptr<DataEntity> Get(index_type idx) const override { return make_data_entity(GetValue(idx)); }
    void Set(size_type idx, std::shared_ptr<DataEntity> const& v) override { Set(idx, data_cast<U>(*v)); }
    void Add(std::shared_ptr<DataEntity> const& v) override { Add(data_cast<U>(*v)); }
    // DataArrayWithType
    virtual value_type GetValue(index_type i) const = 0;
    virtual void Set(size_type idx, value_type const& v) = 0;
    virtual void Add(value_type const& v) = 0;
};

template <typename U>
class DataEntityWrapper<U*> : public DataArrayWithType<U> {
    SP_OBJECT_HEAD(DataEntityWrapper<U*>, DataArrayWithType<U>);
    std::vector<U> m_data_;

   public:
    DataEntityWrapper() {}
    template <typename V>
    DataEntityWrapper(std::initializer_list<V> const& l) {
        for (auto const& v : l) { m_data_.push_back(v); }
    }
    DataEntityWrapper(this_type const& other) : m_data_(other.m_data_) {}
    DataEntityWrapper(this_type&& other) : m_data_(other.m_data_) {}

    virtual ~DataEntityWrapper() {}
    std::vector<U>& get() { return m_data_; }
    std::vector<U> const& get() const { return m_data_; }
    std::ostream& Serialize(std::ostream& os, int indent = 0) const override {
        if (m_data_.size() == 0) { return os; };
        auto it = m_data_.begin();
        os << "[" << *it;
        for (++it; it != m_data_.end(); ++it) { os << "," << *it; }
        os << "]";
        return os;
    }
    // DataEntity

    std::shared_ptr<DataEntity> Duplicate() const override { return std::make_shared<DataEntityWrapper<U*>>(*this); }

    // DataArray
    size_type size() const override { return m_data_.size(); };
    void Delete(size_type idx) override { m_data_.erase(m_data_.begin() + idx); }
    size_type Foreach(std::function<void(std::shared_ptr<DataEntity>)> const& fun) const override {
        for (auto const& item : m_data_) { fun(make_data_entity(item)); }
        return m_data_.size();
    };
    // DataArrayWithType

    U GetValue(index_type idx) const override { return m_data_[idx]; }
    void Set(size_type idx, U const& v) override {
        if (size() < idx) { m_data_.resize(idx); }
        m_data_[idx] = v;
    }
    void Add(U const& v) override { m_data_.push_back(v); }
};
//
// template <typename U, int N>
// class DataEntityWrapper<simpla::algebra::declare::nTuple_<U, N>> : public DataArrayWithType<U> {
//    typedef simpla::algebra::declare::nTuple_<U, N> tuple_type;
//    SP_OBJECT_HEAD(DataEntityWrapper<tuple_type>, DataArrayWithType<U>);
//    tuple_type m_data_;
//
//   public:
//    DataEntityWrapper() {}
//    DataEntityWrapper(tuple_type const& other) : m_data_(other) {}
//    DataEntityWrapper(this_type const& other) : m_data_(other.m_data_) {}
//    DataEntityWrapper(this_type&& other) : m_data_(other.m_data_) {}
//    virtual ~DataEntityWrapper() {}
//    tuple_type& get() { return m_data_; }
//    tuple_type const& get() const { return m_data_; }
//    // DataEntity
//    virtual std::shared_ptr<DataEntity> Duplicate() { return std::make_shared<this_type>(*this); }
//
//    // DataArray
//    virtual size_type size() const { return static_cast<size_type>(N); };
//    virtual size_type Foreach(std::function<void(std::shared_ptr<DataEntity>)> const& fun) const {
//        for (size_type s = 0; s < N; ++s) { fun(make_data_entity(m_data_[s])); }
//        return static_cast<size_type>(N);
//    };
//    virtual void Add(std::shared_ptr<DataEntity> const&) { UNSUPPORTED; };
//    virtual void Delete(size_type idx) { UNSUPPORTED; };
//    // DataArrayWithType
//
//    virtual U GetValue(index_type idx) const { return m_data_[idx]; }
//    virtual void Set(size_type idx, U const& v) {
//        ASSERT(size() > idx);
//        m_data_[idx] = v;
//    }
//    virtual void Add(U const&) { UNSUPPORTED; };
//
//    tuple_type value() const { return m_data_; }
//};
//
// template <typename U, int N0, int N1, int... N>
// class DataEntityWrapper<simpla::algebra::declare::nTuple_<U, N0, N1, N...>> : public DataArray {
//    typedef simpla::algebra::declare::nTuple_<U, N0, N1, N...> tuple_type;
//    SP_OBJECT_HEAD(tuple_type, DataArray);
//    tuple_type m_data_;
//
//   public:
//    DataEntityWrapper() {}
//    DataEntityWrapper(this_type const& other) : m_data_(other.m_data_) {}
//    DataEntityWrapper(this_type&& other) : m_data_(other.m_data_) {}
//
//    virtual ~DataEntityWrapper() {}
//    tuple_type& get() { return m_data_; }
//    tuple_type const& get() const { return m_data_; }
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
//    virtual void Delete(size_type idx) { UNSUPPORTED; };
//    // DataArrayWithType
//    virtual U GetValue(index_type idx) const { return m_data_[idx]; }
//
//    virtual void Set(size_type idx, U const& v) {
//        ASSERT(size() > idx);
//        m_data_[idx] = v;
//    }
//    virtual void Add(U const&) { UNSUPPORTED; };
//};

template <typename U>
std::shared_ptr<DataEntity> make_data_entity(std::initializer_list<U> const& u) {
    return std::make_shared<DataEntityWrapper<U*>>(u);
}
template <typename U>
std::shared_ptr<DataEntity> make_data_entity(std::initializer_list<std::initializer_list<U>> const& u) {
    return std::make_shared<DataEntityWrapper<void*>>(u);
}
inline std::shared_ptr<DataEntity> make_data_entity(std::initializer_list<char const*> const& u) {
    return std::make_shared<DataEntityWrapper<std::string*>>(u);
}
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
//    data_entity_from_helper0(*a.Get(N - 1), std::get<N - 1>(v));
//    data_entity_from_helper(a, v, std::integral_constant<int, N - 1>());
//};
//
// template <typename V>
// void data_entity_to_helper0(V const& src, DataArray& dest, size_type N) {
//    dest.Set(N - 1, data_entity_traits<V>::to(src));
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
