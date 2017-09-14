//
// Created by salmon on 16-10-20.
//

#ifndef SIMPLA_ATTRIBUTEVIEW_H
#define SIMPLA_ATTRIBUTEVIEW_H

#include "simpla/SIMPLA_config.h"

#include "EngineObject.h"
#include "simpla/algebra/Array.h"
#include "simpla/algebra/ArrayNTuple.h"
#include "simpla/data/Data.h"
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

    AttributeGroup(AttributeGroup const &other) = delete;
    AttributeGroup(AttributeGroup &&other) = delete;
    AttributeGroup &operator=(AttributeGroup const &other) = delete;
    AttributeGroup &operator=(AttributeGroup &&other) = delete;

    virtual std::shared_ptr<data::DataNode> Serialize() const;
    virtual void Deserialize(std::shared_ptr<data::DataNode> const &);

    virtual void Push(const std::shared_ptr<data::DataNode> &);
    virtual std::shared_ptr<data::DataNode> Pop() const;

    auto &GetAttributes() { return m_attributes_; }
    auto const &GetAttributes() const { return m_attributes_; }

    void Detach(Attribute *attr);
    void Attach(Attribute *attr);

    std::shared_ptr<data::DataNode> RegisterAttributes();
    std::shared_ptr<data::DataNode> GetAttributeDescription(std::string const &k) const;
    //    std::shared_ptr<data::DataNode> GetDescriptions() const;

    //    virtual void RegisterAt(AttributeGroup *);
    //    virtual void DeregisterFrom(AttributeGroup *);
    //    virtual void RegisterDescription(std::map<std::string, std::shared_ptr<AttributeDesc>> *) const;
    //    bool has(std::string const &k) const;
    //    bool check(std::string const &k, std::type_info const &t_info) const;
    //
    //    Attribute *GetPatch(std::string const &k);
    //    Attribute const *GetPatch(std::string const &k) const;
    //    std::map<std::string, Attribute *> const &GetAll() const;
    //    virtual std::string GetDomainPrefix() const { return ""; }
    //
    //    template <typename T>
    //    T GetAttribute(std::string const &k) const;

   private:
    std::map<std::string, Attribute *> m_attributes_;
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
    static std::string FancyTypeName() { return __STRING(Attribute); }
    std::string TypeName() const override { return simpla::traits::type_name<this_type>::value(); }

    static bool _is_registered;

   private:
    typedef EngineObject base_type;
    typedef Attribute this_type;
    struct pimpl_s;
    pimpl_s *m_pimpl_ = nullptr;

   public:
    Attribute();
    ~Attribute() override;

    template <typename THost, typename... Args>
    explicit Attribute(THost host, Args &&... args) : Attribute() {
        Register(host);
        db()->SetValue(std::forward<Args>(args)...);
    };

    void ReRegister(std::shared_ptr<Attribute> const &) const;

    void Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) override;
    std::shared_ptr<simpla::data::DataNode> Serialize() const override;
    void DoSetUp() override;
    void DoUpdate() override;
    void DoTearDown() override;

    virtual std::type_info const &value_type_info() const = 0;
    virtual int GetIFORM() const = 0;
    virtual int GetDOF(int) const = 0;
    virtual int GetRank() const = 0;
    virtual void SetDOF(int rank, int const *d) = 0;

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
    typedef nTuple<Array<V>, (IFORM == NODE || IFORM == CELL) ? 1 : 3, DOF...> data_type;
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
    static std::string FancyTypeName();
    std::string TypeName() const override { return simpla::traits::type_name<this_type>::value(); }

    static bool _is_registered;

   private:
    typedef Attribute base_type;
    typedef AttributeT this_type;
    typedef typename attribute_traits<V, IFORM, DOF...>::data_type data_type;
    typedef V value_type;
    typedef Array<value_type> array_type;

   public:
    static constexpr int iform = IFORM;

    AttributeT();

    template <typename... Args>
    explicit AttributeT(Args &&... args) : Attribute(std::forward<Args>(args)...) {}

    ~AttributeT() override;

    template <typename... Args>
    static std::shared_ptr<this_type> New(Args &&... args) {
        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...));
    }

    static std::shared_ptr<this_type> New(std::shared_ptr<simpla::data::DataNode> cfg) {
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
    int GetRank() const override { return sizeof...(DOF); };
    void SetDOF(int rank, int const *d) override { DOMAIN_ERROR; };

    auto &GetData(int n) { return data_type::operator[](n); }
    auto const &GetData(int n) const { return data_type::operator[](n); }

    template <typename... Args>
    auto &Get(index_type i0, Args &&... args) {
        return GetData(i0).at(std::forward<Args>(args)...);
    }
    template <typename... Args>
    auto const &Get(index_type i0, Args &&... args) const {
        return GetData(i0).at(std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto &at(index_type n0, Args &&... args) {
        return Get(n0, std::forward<Args>(args)...);
    }
    template <typename... Args>
    auto const &at(index_type n0, Args &&... args) const {
        return Get(n0, std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto &operator()(index_type n0, Args &&... args) {
        return Get(n0, std::forward<Args>(args)...);
    }
    template <typename... Args>
    auto const &operator()(index_type n0, Args &&... args) const {
        return Get(n0, std::forward<Args>(args)...);
    }

    template <typename RHS>
    void Assign(RHS const &rhs);
    template <typename RHS>
    this_type &operator=(RHS const &rhs) {
        Assign(rhs);
        return *this;
    }

   private:
    static constexpr int m_extents_[sizeof...(DOF) + 1] = {(IFORM == NODE || IFORM == CELL) ? 1 : 3, DOF...};
};
template <typename V, int IFORM, int... DOF>
constexpr int AttributeT<V, IFORM, DOF...>::m_extents_[sizeof...(DOF) + 1];

template <typename V, int IFORM, int... DOF>
AttributeT<V, IFORM, DOF...>::AttributeT(){};
template <typename V, int IFORM, int... DOF>
AttributeT<V, IFORM, DOF...>::~AttributeT(){};
template <typename V, int IFORM, int... DOF>
std::string AttributeT<V, IFORM, DOF...>::FancyTypeName() {
    return "AttributeT<" + simpla::traits::type_name<V>::value() + "," + EntityIFORMName[IFORM] +
           ((sizeof...(DOF) == 0) ? "" : ("," + simpla::traits::to_string(DOF...))) + ">";
    ;
}

template <typename V, int IFORM, int... DOF>
std::shared_ptr<data::DataNode> AttributeT<V, IFORM, DOF...>::Serialize() const {
    auto res = base_type::Serialize();
    res->Set("_DATA_", Pop());
    return res;
};
template <typename V, int IFORM, int... DOF>
void AttributeT<V, IFORM, DOF...>::Deserialize(std::shared_ptr<data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    Push(cfg->Get("_DATA_"));
};

namespace detail {
template <typename U>
std::shared_ptr<data::DataNode> pop_data(Array<U> const &v) {
    auto d = data::DataBlockT<U>::New();
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
    } else if (auto p = std::dynamic_pointer_cast<data::DataBlockT<U>>(src->GetEntity())) {
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
void AttributeT<V, IFORM, DOF...>::Push(const std::shared_ptr<data::DataNode> &d) {
    detail::push_data(*this, d);
};

template <typename V, int IFORM, int... DOF>
std::shared_ptr<data::DataNode> AttributeT<V, IFORM, DOF...>::Pop() const {
    return detail::pop_data(*this);
};
template <typename V, int IFORM, int... DOF>
template <typename RHS>
void AttributeT<V, IFORM, DOF...>::Assign(RHS const &rhs){

};
}  // namespace engine
}  // namespace simpla

#endif  // SIMPLA_ATTRIBUTEVIEW_H
