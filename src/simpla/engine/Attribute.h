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
    SP_OBJECT_HEAD(Attribute, SPObject)

   protected:
    template <typename THost, typename... Args>
    explicit Attribute(THost host, Args &&... args) : Attribute() {
        Register(host);
        db()->SetValue(std::forward<Args>(args)...);
    };

    void ReRegister(std::shared_ptr<Attribute> const &) const;

   public:
    void Push(const std::shared_ptr<data::DataNode> &) override;
    std::shared_ptr<data::DataNode> Pop() const override;

    virtual std::type_info const &value_type_info() const = 0;
    virtual int GetIFORM() const = 0;
    virtual int GetDOF(int) const = 0;
    virtual int GetRank() const = 0;
    virtual void SetDOF(int rank, int const *d) = 0;

    template <typename THost>
    void Register(THost p) {}
    void Register(AttributeGroup *p = nullptr);
    void Deregister(AttributeGroup *p = nullptr);

    virtual size_type CopyOut(Attribute &other) const;
    virtual size_type CopyIn(Attribute const &other);
    virtual std::shared_ptr<Attribute> Duplicate() const;

    virtual bool isNull() const;
    virtual bool empty() const { return isNull(); };
    virtual void Clear();
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
    SP_OBJECT_HEAD(AttributeT, Attribute)

    typedef V value_type;
    typedef Array<value_type> array_type;
    typedef typename attribute_traits<V, IFORM, DOF...>::data_type data_type;
    static constexpr int iform = IFORM;

   public:
    template <typename... Args>
    explicit AttributeT(Args &&... args) : Attribute(std::forward<Args>(args)...) {}

    void Push(const std::shared_ptr<data::DataNode> &) override;
    std::shared_ptr<data::DataNode> Pop() const override;

    std::shared_ptr<Attribute> Duplicate() const override {
        std::shared_ptr<this_type> res(new this_type);
        ReRegister(res);
        return res;
    }

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
    if (auto p = std::dynamic_pointer_cast<data::DataBlockT<U>>(src->GetEntity())) {
        Array<U>(*p).swap(dest);
        count = 1;
    }

    return count;
}
template <typename U, int N0, int... N>
size_type push_data(nTuple<Array<U>, N0, N...> &v, std::shared_ptr<data::DataNode> const &src) {
    size_type count = 0;
    for (int i = 0; i < N0; ++i) { count += push_data(v[i], src->Get(i)); }
    return count;
}
}  // namespace detail{

template <typename V, int IFORM, int... DOF>
void AttributeT<V, IFORM, DOF...>::Push(const std::shared_ptr<data::DataNode> &d) {
    detail::push_data(*this, d);
};

template <typename V, int IFORM, int... DOF>
std::shared_ptr<data::DataNode> AttributeT<V, IFORM, DOF...>::Pop() const {
    return detail::pop_data(*this);
};

//
// template <typename U, typename... Others, int... N>
// void Attribute::PushData(nTuple<Array<U, Others...>, N...> *d) {
//    typedef Array<U, Others...> array_type;
//    auto *blk = dynamic_cast<data::DataMultiArray<array_type> *>(GetDataBlock());
//    if (blk != nullptr) {
//        int count = 0;
//        traits::foreach (*d, [&](array_type &a, auto &&... idx) {
//            array_type(*blk->GetEntity(count)).swap(a);
//            ++count;
//        });
//    }
//    Tag();
//};
// template <typename U, typename... Others, int... N>
// void Attribute::PopData(nTuple<Array<U, Others...>, N...> *d) {
//    typedef Array<U, Others...> array_type;
//    auto *blk = dynamic_cast<data::DataMultiArray<array_type> *>(GetDataBlock());
//    if (blk == nullptr) {
//        Push(std::make_shared<data::DataMultiArray<array_type>>(d->size()));
//        blk = dynamic_cast<data::DataMultiArray<array_type> *>(GetDataBlock());
//    }
//    int count = 0;
//    traits::foreach (*d, [&](array_type &a, auto &&... idx) {
//        array_type(a).swap(*blk->GetEntity(count));
//        a.reset();
//        ++count;
//    });
//    ResetTag();
//};

// template <typename T>
// T AttributeGroup::GetAttribute(std::string const &k) const {
//    return T(AttributeGroup::GetPatch(k)->cast_as<T>());
//};
//
// template <typename, typename Enable = void>
// class AttributeViewAdapter {};
//
// CHECK_MEMBER_TYPE(mesh_type, mesh_type);
// CHECK_MEMBER_FUNCTION(has_swap, swap)
// template <typename U>
// class AttributeViewAdapter<
//    U, std::enable_if_t<std::is_copy_constructible<U>::value && traits::has_swap<U, void(U &)>::value>>
//    : public Attribute, public U {
//    SP_DEFINE_FANCY_TYPE_NAME(AttributeViewAdapter<U>, Attribute);
//
//    typedef algebra::traits::value_type_t<U> value_type;
//    typedef typename algebra::traits::mesh_type_t<U> mesh_type;
//    static const int iform = algebra::traits::iform<U>::value;
//    static const int dof = algebra::traits::dof<U>::value;
//    static const int NDIMS = algebra::traits::ndims<U>::value;
//    static const int num_of_sub = algebra::traits::num_of_sub<U>::value;
//
//   public:
//    typedef std::true_type prefer_pass_by_reference;
//
//    template <typename... Args>
//    explicit AttributeViewAdapter(AttributeGroup *b, Args &&... args)
//        : Attribute(b, data::make_data(std::forward<Args>(args)...)) {}
//
//    AttributeViewAdapter(AttributeViewAdapter &&) = delete;
//    AttributeViewAdapter(AttributeViewAdapter const &) = delete;
//    virtual ~AttributeViewAdapter() {}
//
//    virtual int GetIFORM() const { return iform; };
//    virtual int GetDOF() const { return dof; };
//    virtual std::type_info const &value_type_info() const { return typeid(value_type); };  //!< value type
//    virtual std::type_info const &mesh_type_info() const { return typeid(void); };         //!< mesh type
//    virtual void Clear() { U::Clear(); }
//    virtual void SetBaseMesh(MeshBase const *){};
//    virtual MeshBase const *GetBaseMesh() const { return nullptr; };
//    virtual void GetPatch(std::shared_ptr<MeshBlock> const &m, std::shared_ptr<data::DataNode> const
//    &d) {
//        data::data_cast<U>(*d).swap(*this);
//    };
//    virtual std::pair<std::shared_ptr<MeshBlock>, std::shared_ptr<data::DataNode>> Serialize() {
//        return std::make_pair(std::shared_ptr<MeshBlock>(nullptr), data::make_data(*this));
//    };
//    template <typename TExpr>
//    this_type &operator=(TExpr const &expr) {
//        Click();
//        U::operator=(expr);
//        return *this;
//    };
//
//    bool UpdatePatch() final {
//        if (!Attribute::UpdatePatch()) { return false; }
//        return U::UpdatePatch();
//    }
//};
//
// template <typename TV, typename TM, int IFORM = NODE, int DOF = 1>
// using FieldAttribute = Field<TV, TM, IFORM, DOF>;
//
// template <typename TV = Real, int IFORM = NODE, int DOF = 1>
// using DataAttribute = AttributeViewAdapter<Array<TV, 3 + (((IFORM == NODE || IFORM == CELL) && DOF == 1) ? 0 :
// 1)>>;
//
// template <typename TV, int IFORM = NODE, int DOF = 1>
// struct DataAttribute : public Attribute,
//                       public Array<TV, 3 + (((IFORM == NODE || IFORM == CELL) && DOF == 1) ? 0 : 1)> {
//    typedef Array<TV, 3 + (((IFORM == NODE || IFORM == CELL) && DOF == 1) ? 0 : 1)> array_type;
//    typedef DataAttribute<TV, IFORM, DOF> data_attr_type;
//    SP_DEFINE_FANCY_TYPE_NAME(data_attr_type, Attribute);
//    CHOICE_TYPE_WITH_TYPE_MEMBER(mesh_traits, mesh_type, MeshBase)
//    typedef TV value_type;
//    static constexpr int GetIFORM = IFORM;
//    static constexpr int GetDOF = DOF;
//    typedef MeshBase mesh_type;
//
//    template <typename TM, typename... Args>
//    DataAttribute(TM *w, Args &&... args)
//        : base_type(w, AttributeDesc::create<value_type, GetIFORM, GetDOF>(std::forward<Args>(args)...)),
//          Attribute(<#initializer #>, nullptr, <#initializer #>) {}
//    template <typename TM>
//    DataAttribute(TM *m, std::initializer_list<data::KeyValue> const &param)
//        : base_type(m, AttributeDesc::create<value_type, GetIFORM, GetDOF>(param)),
//          Attribute(<#initializer #>, nullptr, <#initializer #>) {}
//    DataAttribute(DataAttribute &&) = delete;
//    DataAttribute(DataAttribute const &) = delete;
//    virtual ~DataAttribute() {}
//
//    template <typename... Args>
//    static std::shared_ptr<this_type> CreateNew(Args &&... args) {
//        return std::make_shared<this_type>(std::forward<Args>(args)...);
//    }
//
//    virtual std::shared_ptr<DataBlock> InitialCondition(void *p = nullptr) const {
//        std::shared_ptr<value_type> d(nullptr);
//        if (p != nullptr) {
//            d = std::shared_ptr<value_type>(static_cast<value_type *>(p), simpla::tags::do_nothing());
//        } else {
//#ifdef USE_MEMORYPOOL
//            d = sp_alloc_array<value_type>(array_type::size());
//#else
//            d = std::shared_ptr<value_type>(new value_type[array_type::size()]);
//#endif
//        }
//        return std::dynamic_pointer_cast<DataBlock>(
//            std::make_shared<DefaultDataBlock<value_type, IFORM, DOF>>(d, array_type::size()));
//    };
//
//    using array_type::operator=;
//    template <typename... Args>
//    static std::shared_ptr<this_type> make_shared(Args &&... args) {
//        return std::make_shared<this_type>(std::forward<Args>(args)...);
//    }
//    static std::shared_ptr<this_type> make_shared(MeshBase *c, std::initializer_list<data::KeyValue> const &param) {
//        return std::make_shared<this_type>(c, param);
//    }
//    virtual std::ostream &Print(std::ostream &os, int indent = 0) const { return array_type::Print(os, indent); }
//
//    virtual value_type *data() { return reinterpret_cast<value_type *>(Attribute::GetDataBlock()->raw_data()); }
//
//    virtual void UpdatePatch() {
//        Attribute::UpdatePatch();
//        array_type::UpdatePatch();
//    }
//    virtual void DoFinalize() {
//        array_type::DoFinalize();
//        Attribute::DoFinalize();
//    }
//
//    virtual void Clear() { array_type::Clear(); }
//};

}  // namespace engine
}  // namespace simpla

#endif  // SIMPLA_ATTRIBUTEVIEW_H
