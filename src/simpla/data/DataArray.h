//
// Created by salmon on 17-3-8.
//

#ifndef SIMPLA_DATAARRAY_H
#define SIMPLA_DATAARRAY_H
#include <functional>
#include "DataNode.h"
#include "DataTraits.h"
namespace simpla {
namespace data {
struct DataArray : public DataNode {
    SP_OBJECT_HEAD(DataArray, DataNode)

    struct pimpl_s;
    pimpl_s* m_pimpl_ = nullptr;

   protected:
    DataArray() = default;

   public:
    ~DataArray() override = default;
    //    SP_DEFAULT_CONSTRUCT(DataArray)

    static std::shared_ptr<DataArray> New();

    /** @addtogroup{ capacity */
    virtual bool isNull() { return true; }
    /** @} */
    /** @addtogroup{ access */
    virtual std::shared_ptr<DataNode> Root() { return FindNode("/", true); }
    virtual std::shared_ptr<DataNode> Parent() const { return FindNode("..", true); }

    virtual std::shared_ptr<DataNode> FirstChild() const { return nullptr; }
    virtual std::shared_ptr<DataNode> Next() const { return nullptr; }

    virtual std::shared_ptr<DataNode> NewNode(std::string const& uri, bool recursive = false) { return nullptr; };

    virtual std::shared_ptr<DataNode> FindNode(std::string const& uri, bool recursive = false) const {
        return nullptr;
    };

    virtual size_type GetNumberOfChildren() const { return 0; }
    virtual std::shared_ptr<DataNode> GetNodeByIndex(index_type idx) const { return nullptr; }
    virtual std::shared_ptr<DataNode> GetNodeByName(std::string const& s) const { return nullptr; }
    /** @} */

    /** @addtogroup{  modify */
    virtual int SetNodeByIndex(index_type idx, std::shared_ptr<DataNode> const& v) { return 0; }
    virtual int SetNodeByName(std::string const& k, std::shared_ptr<DataNode> const& v) { return 0; }
    virtual std::shared_ptr<DataNode> AddNode(std::shared_ptr<DataNode> const& v = nullptr) { return nullptr; }
    /** @}  */

    /** @addtogroup{ */
};

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
