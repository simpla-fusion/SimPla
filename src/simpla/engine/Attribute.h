//
// Created by salmon on 16-10-20.
//

#ifndef SIMPLA_ATTRIBUTE_H
#define SIMPLA_ATTRIBUTE_H

#include "simpla/SIMPLA_config.h"

#include "EngineObject.h"
#include "simpla/algebra/Array.h"
#include "simpla/algebra/ExpressionTemplate.h"
#include "simpla/data/Data.h"
#include "simpla/utilities/SPDefines.h"
#include "simpla/utilities/type_traits.h"

namespace simpla {
template <typename V, typename SFC>
class Array;
namespace engine {
class Attribute;

///**
// *  permissions
// *
// *   r : readable
// *   w : writable
// *   c : create/delete
// *
// * [ 0:false 1:true ]
// * 0b0 0 0 0 0
// *   | | | | |------: is shared between different domain
// *   | | | |--------: has ghost cell
// *   | | |----------: PERSISTENT, if false then destroy data when Attribute is destructed
// *   | |------------: become unmodifiable after first write
// *   |--------------: is coordinate
// */
// enum AttributeTag {
//    SCRATCH = 0,
//    SHARED = 1,            //
//    GHOSTED = 1 << 1,      //
//    PERSISTENT = 1 << 2,   //
//    INPUT = 1 << 3,        //  can only be written once
//    COORDINATES = 1 << 4,  //  coordinate of mesh vertex
//    NO_FILL,
//    GLOBAL = SHARED | GHOSTED | PERSISTENT,
//    PRIVATE = GHOSTED | PERSISTENT,
//    DEFAULT_ATTRIBUTE_TAG = GLOBAL
//};

class AttributeGroup {
   public:
    typedef Attribute attribute_type;
    AttributeGroup();
    virtual ~AttributeGroup();

    virtual std::shared_ptr<data::DataNode> Serialize() const;
    virtual void Deserialize(std::shared_ptr<data::DataNode> const &);

    virtual void Push(const std::shared_ptr<data::DataNode> &);
    virtual std::shared_ptr<data::DataNode> Pop() const;

    std::set<Attribute *> &GetAttributes();
    std::set<Attribute *> const &GetAttributes() const;

    void Detach(Attribute *attr);
    void Attach(Attribute *attr);

    std::shared_ptr<data::DataNode> RegisterAttributes();

   private:
    struct pimpl_s;
    pimpl_s *m_pimpl_ = nullptr;
};

/**
 *
 * Attribute
 *
 * @startuml
 * title Life cycle
 * actor Main
 * participant AttributeView
 * participant AttributeViewBundle
 * participant DomainView
 * participant AttributeViewAdapter as AttributeT <<T,IFORM,DOF>>
 * participant Attribute
 * Main->AttributeView: CreateDataBlock()
 * activate AttributeView
 *  AttributeView->AttributeT: CreateDataBlock(p)
 *  activate AttributeT
 *      AttributeT -> Mesh:
 *      Mesh --> AttributeT:
 *      AttributeT --> AttributeView :return DataBlock
 *  deactivate AttributeT
 *
 * AttributeT-->Main: return DataBlock
 * deactivate AttributeView
 * @enduml
 *
 *
 */
struct Attribute : public EngineObject {
   public:
    static std::string FancyTypeName() { return "Attribute"; }
    std::string TypeName() const override { return "Attribute"; }

    static bool _is_registered;

   private:
    typedef EngineObject base_type;
    typedef Attribute this_type;
    struct pimpl_s;
    pimpl_s *m_pimpl_ = nullptr;

   public:
    Attribute();
    ~Attribute() override;
    Attribute(this_type const &other) = delete;  // { UNIMPLEMENTED; };
    Attribute(this_type &&other) = delete;       // { UNIMPLEMENTED; };

    template <typename THost, typename... Args>
    explicit Attribute(THost host, Args &&... args) : Attribute() {
        Register(host);
        db()->SetValue(std::forward<Args>(args)...);
    };
    static std::shared_ptr<this_type> New(std::shared_ptr<simpla::data::DataNode> const &cfg);

    void ReRegister(std::shared_ptr<Attribute> const &) const;

    void Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) override;
    std::shared_ptr<simpla::data::DataNode> Serialize() const override;
    void DoSetUp() override;
    void DoUpdate() override;
    void DoTearDown() override;

    virtual std::type_info const &value_type_info() const = 0;
    virtual int GetIFORM() const = 0;
    virtual int GetDOF(int) const = 0;
    virtual int const *GetDOFs() const = 0;
    virtual int GetRank() const = 0;
    virtual void SetDOF(int rank, int const *d) = 0;
    virtual std::shared_ptr<data::DataNode> GetDescription() const = 0;

    void Register(AttributeGroup *p = nullptr);
    void Deregister(AttributeGroup *p = nullptr);

    void Push(const std::shared_ptr<data::DataNode> &) override = 0;
    std::shared_ptr<data::DataNode> Pop() const override = 0;

    virtual std::shared_ptr<Attribute> Duplicate() const = 0;

    virtual bool isNull() const = 0;
    virtual bool empty() const { return isNull(); };
    virtual void Clear() = 0;
};

template <typename V, int IFORM, int... DOF>
struct attribute_traits {
    typedef std::conditional_t<(IFORM == EDGE || IFORM == FACE), nTuple<Array<V>, 3, DOF...>,
                               std::conditional_t<sizeof...(DOF) == 0, Array<V>, nTuple<Array<V>, DOF...>>>
        data_type;
};
template <typename V>
struct attribute_traits<V, NODE> {
    typedef Array<V> data_type;
};
template <typename V>
struct attribute_traits<V, CELL> {
    typedef Array<V> data_type;
};

template <typename V, int IFORM, int... DOF>
struct AttributeT : public Attribute, public attribute_traits<V, IFORM, DOF...>::data_type {
   public:
    static bool _is_registered;

   private:
    typedef Attribute base_type;
    typedef AttributeT this_type;

   public:
    typedef typename attribute_traits<V, IFORM, DOF...>::data_type data_type;
    typedef V value_type;
    typedef Array<value_type> array_type;

    static constexpr int iform = IFORM;

    AttributeT();

    template <typename... Args>
    explicit AttributeT(Args &&... args) : Attribute(std::forward<Args>(args)...) {}

    ~AttributeT() override;

    template <typename... Args>
    static std::shared_ptr<this_type> New(Args &&... args) {
        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...));
    }

    static std::shared_ptr<this_type> New(std::shared_ptr<simpla::data::DataNode> const &cfg) {
        auto res = std::shared_ptr<this_type>(new this_type());
        res->Deserialize(cfg);
        return res;
    }

    void Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) override;
    std::shared_ptr<simpla::data::DataNode> Serialize() const override;
    void DoSetUp() override;
    void DoUpdate() override;
    void DoTearDown() override;

    void Push(const std::shared_ptr<data::DataNode> &) override;
    std::shared_ptr<data::DataNode> Pop() const override;

    std::shared_ptr<Attribute> Duplicate() const override {
        std::shared_ptr<this_type> res(new this_type);
        ReRegister(res);
        return res;
    }
    bool isNull() const override;
    bool empty() const override { return isNull(); };
    void Clear() override;

    std::type_info const &value_type_info() const override { return typeid(V); };
    int GetIFORM() const override { return IFORM; };
    int GetDOF(int n) const override { return m_extents_[n]; };
    int const *GetDOFs() const override { return m_extents_; };
    int GetRank() const override { return sizeof...(DOF); };
    void SetDOF(int rank, int const *d) override { DOMAIN_ERROR; };
    std::shared_ptr<data::DataNode> GetDescription() const override;

    auto &GetData(int n) { return traits::index(dynamic_cast<data_type &>(*this), n); }
    auto const &GetData(int n) const { return traits::index(dynamic_cast<data_type const &>(*this), n); }

    template <typename... Args>
    auto &Get(index_type i0, Args &&... args) {
        return traits::invoke(traits::index(dynamic_cast<data_type &>(*this), i0), std::forward<Args>(args)...);
    }
    template <typename... Args>
    auto const &Get(index_type i0, Args &&... args) const {
        return traits::invoke(traits::index(dynamic_cast<data_type const &>(*this), i0), std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto &at(Args &&... args) {
        return Get(std::forward<Args>(args)...);
    }
    template <typename... Args>
    auto const &at(Args &&... args) const {
        return Get(std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto &operator()(index_type i0, Args &&... args) {
        return Get(i0, std::forward<Args>(args)...);
    }
    template <typename... Args>
    auto const &operator()(index_type i0, Args &&... args) const {
        return Get(i0, std::forward<Args>(args)...);
    }

    template <typename RHS>
    void Assign(RHS const &rhs);
    template <typename RHS>
    this_type &operator=(RHS const &rhs) {
        Assign(rhs);
        return *this;
    }
    template <typename... RHS>
    this_type &operator=(Expression<RHS...> const &rhs) {
        data_type::operator=(rhs);
        return *this;
    }

   private:
    static constexpr int m_extents_[sizeof...(DOF) + 1] = {(IFORM == NODE || IFORM == CELL) ? 1 : 3, DOF...};
};
template <typename V, int IFORM, int... DOF>
constexpr int AttributeT<V, IFORM, DOF...>::m_extents_[sizeof...(DOF) + 1];

template <typename V, int IFORM, int... DOF>
AttributeT<V, IFORM, DOF...>::AttributeT() = default;
template <typename V, int IFORM, int... DOF>
AttributeT<V, IFORM, DOF...>::~AttributeT() = default;

template <typename V, int IFORM, int... DOF>
std::shared_ptr<data::DataNode> AttributeT<V, IFORM, DOF...>::GetDescription() const {
    auto res = data::DataNode::New(data::DataNode::DN_TABLE);
    res->Set(db());
    res->SetValue("Name", GetName());
    res->SetValue("IFORM", IFORM);
    res->SetValue("DOF", DOF...);
    res->SetValue("ValueType", traits::type_name<V>::value());
    return res;
};

namespace detail {
template <typename U>
std::shared_ptr<data::DataNode> pop_data(Array<U> const &v) {
    auto d = data::DataBlock<U>::New();
    Array<U>(v).swap(*d);
    return data::DataNode::New(d);
}
template <typename U, int N0, int... N>
std::shared_ptr<data::DataNode> pop_data(nTuple<Array<U>, N0, N...> const &v) {
    auto res = data::DataNode::New(data::DataNode::DN_ARRAY);
    for (int i = 0; i < N0; ++i) { res->Add(pop_data(v[i])); }
    return res;
}

template <typename U>
size_type push_data(Array<U> &dest, std::shared_ptr<data::DataNode> const &src) {
    size_type count = 0;
    if (src == nullptr) {
    } else if (auto p = std::dynamic_pointer_cast<Array<U>>(src->GetEntity())) {
        Array<U>(*p).swap(dest);
        count = 1;
    }

    return count;
}
template <typename U, int N0, int... N>
size_type push_data(nTuple<Array<U>, N0, N...> &v, std::shared_ptr<data::DataNode> const &src) {
    size_type count = 0;

    for (int i = 0; i < N0; ++i) { count += push_data(v[i], src == nullptr ? nullptr : src->Get(i)); }

    return count;
}

template <typename U>
bool is_null(Array<U> const &d) {
    return d.isNull();
}
template <typename U, int N0, int... N>
bool is_null(nTuple<Array<U>, N0, N...> const &v) {
    bool res = false;
    for (int i = 0; i < N0; ++i) { res = res || is_null(v[i]); }
    return res;
}
template <typename U>
void clear(Array<U> &d) {
    d.Clear();
}
template <typename U, int N0, int... N>
void clear(nTuple<Array<U>, N0, N...> &v) {
    for (int i = 0; i < N0; ++i) { clear(v[i]); }
}
template <typename U>
void update(Array<U> &d) {
    d.alloc();
}
template <typename U, int N0, int... N>
void update(nTuple<Array<U>, N0, N...> &v) {
    for (int i = 0; i < N0; ++i) { update(v[i]); }
}
}  // namespace detail{

template <typename V, int IFORM, int... DOF>
void AttributeT<V, IFORM, DOF...>::DoSetUp(){};
template <typename V, int IFORM, int... DOF>
void AttributeT<V, IFORM, DOF...>::DoUpdate() {
    detail::update(*this);
};
template <typename V, int IFORM, int... DOF>
void AttributeT<V, IFORM, DOF...>::DoTearDown(){};
template <typename V, int IFORM, int... DOF>
bool AttributeT<V, IFORM, DOF...>::isNull() const {
    return detail::is_null(*this);
};
template <typename V, int IFORM, int... DOF>
void AttributeT<V, IFORM, DOF...>::Clear() {
    Update();
    detail::clear(*this);
};
template <typename V, int IFORM, int... DOF>
std::shared_ptr<data::DataNode> AttributeT<V, IFORM, DOF...>::Serialize() const {
    auto res = base_type::Serialize();
    res->Set(GetDescription());
    return res;
};
template <typename V, int IFORM, int... DOF>
void AttributeT<V, IFORM, DOF...>::Deserialize(std::shared_ptr<data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
};
template <typename V, int IFORM, int... DOF>
void AttributeT<V, IFORM, DOF...>::Push(const std::shared_ptr<data::DataNode> &d) {
    detail::push_data(*this, d);
};

template <typename V, int IFORM, int... DOF>
std::shared_ptr<data::DataNode> AttributeT<V, IFORM, DOF...>::Pop() const {
    return detail::pop_data(*this);
};

namespace detail {
template <size_type I, typename U>
U const &get(U const &u) {
    return u;
}
template <size_type I, typename U, int... N>
auto const &get(nTuple<U, N...> const &u) {
    return u[I];
}

template <size_type I, typename U>
U &get(U &u) {
    return u;
}
template <size_type I, typename U, int... N>
auto &get(nTuple<U, N...> &u) {
    return u[I];
}
template <typename U, typename... Idx>
auto get_value(std::true_type, U const &u, Idx &&... idx) {
    return u(std::forward<Idx>(idx)...);
}
template <typename U, typename... Idx>
auto get_value(std::false_type, U const &u, Idx &&... idx) {
    return u;
}

template <typename U, typename... Idx>
auto get_value(U const &u, Idx &&... idx) {
    return get_value(std::integral_constant<bool, traits::is_invocable<U, Idx...>::value>(), u,
                     std::forward<Idx>(idx)...);
}
template <typename... V, typename U>
void Assign_(Array<V...> &f, U const &v) {
    f = v;
};

template <typename LHS, typename RHS>
void Assign_(std::integer_sequence<size_type>, LHS &lhs, RHS const &rhs){};

template <size_type I0, size_type... I, typename LHS, typename RHS>
void Assign_(std::integer_sequence<size_type, I0, I...>, LHS &lhs, RHS const &rhs) {
    Assign_(get<I0>(lhs), [&](index_type x, index_type y, index_type z) { return get<I0>(get_value(rhs, x, y, z)); });
    Assign_(std::integer_sequence<size_type, I...>(), lhs, rhs);
};

template <typename V, int N0, int... N, typename U>
void Assign_(nTuple<V, N0, N...> &lhs, U const &rhs) {
    Assign_(std::make_index_sequence<N0>(), lhs, rhs);
};
template <typename V, int... N, typename RHS>
void Assign(AttributeT<V, NODE, N...> &lhs, RHS const &rhs) {
    Assign_(lhs, rhs);
};
template <typename V, int... N, typename RHS>
void Assign(AttributeT<V, CELL, N...> &lhs, RHS const &rhs) {
    Assign_(lhs, rhs);
};
template <typename V, int... DOF, typename RHS>
void Assign(AttributeT<V, EDGE, DOF...> &lhs, RHS const &rhs) {
    Assign_(lhs[0], [&](auto &&... idx) { return get_value(rhs, 0b001, std::forward<decltype(idx)>(idx)...); });
    Assign_(lhs[1], [&](auto &&... idx) { return get_value(rhs, 0b010, std::forward<decltype(idx)>(idx)...); });
    Assign_(lhs[2], [&](auto &&... idx) { return get_value(rhs, 0b100, std::forward<decltype(idx)>(idx)...); });
};
template <typename V, int... DOF, typename RHS>
void Assign(AttributeT<V, FACE, DOF...> &lhs, RHS const &rhs) {
    Assign_(lhs[0], [&](auto &&... idx) { return get_value(rhs, 0b110, std::forward<decltype(idx)>(idx)...); });
    Assign_(lhs[1], [&](auto &&... idx) { return get_value(rhs, 0b101, std::forward<decltype(idx)>(idx)...); });
    Assign_(lhs[2], [&](auto &&... idx) { return get_value(rhs, 0b011, std::forward<decltype(idx)>(idx)...); });
};
}  // namespace detail{

template <typename V, int IFORM, int... DOF>
template <typename RHS>
void AttributeT<V, IFORM, DOF...>::Assign(RHS const &rhs) {
    detail::Assign(*this, rhs);
};

}  // namespace engine

namespace traits {
template <typename>
struct reference;
template <typename TV, int... I>
struct reference<engine::AttributeT<TV, I...>> {
    typedef const engine::AttributeT<TV, I...> &type;
};

template <typename TV, int... I>
struct reference<const engine::AttributeT<TV, I...>> {
    typedef const engine::AttributeT<TV, I...> &type;
};
template <typename>
struct iform;

template <typename T>
struct iform<const T> : public std::integral_constant<int, iform<T>::value> {};
template <typename TV, int IFORM, int... DOF>
struct iform<engine::AttributeT<TV, IFORM, DOF...>> : public std::integral_constant<int, IFORM> {};
template <typename TF>
struct dof;
template <typename TV, int IFORM, int... DOF>
struct dof<engine::AttributeT<TV, IFORM, DOF...>>
    : public std::integral_constant<int, reduction_v(tags::multiplication(), 1, DOF...)> {};

template <typename TV, int... DOF>
struct value_type<engine::AttributeT<TV, DOF...>> {
    typedef TV type;
};
}  // namespace traits {
}  // namespace simpla

namespace std {
template <typename V, int... N>
struct rank<simpla::engine::AttributeT<V, N...>>
    : public integral_constant<size_t, rank<typename simpla::engine::AttributeT<V, N...>::data_type>::value> {};

template <typename V, int... N, unsigned I>
struct extent<simpla::engine::AttributeT<V, N...>, I>
    : public integral_constant<size_t, extent<typename simpla::engine::AttributeT<V, N...>::data_type, I>::value> {};
}
#endif  // SIMPLA_ATTRIBUTE_H
