//
// Created by salmon on 16-10-20.
//

#ifndef SIMPLA_ATTRIBUTEVIEW_H
#define SIMPLA_ATTRIBUTEVIEW_H

#include "simpla/SIMPLA_config.h"

#include "simpla/data/Data.h"
#include "simpla/utilities/Signal.h"
#include "simpla/utilities/type_traits.h"

#include "EngineObject.h"
#include "MeshBlock.h"

namespace simpla {
template <typename V, typename SFC>
class Array;
namespace engine {
class DomainBase;
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

    auto &GetAttributes() { return m_attributes_; }
    auto const &GetAttributes() const { return m_attributes_; }

    virtual int Push(const std::shared_ptr<data::DataNode> &);
    virtual std::shared_ptr<data::DataNode> Pop();

    void Detach(Attribute *attr);
    void Attach(Attribute *attr);

    std::shared_ptr<data::DataNode> RegisterAttributes();

    std::shared_ptr<data::DataNode> GetAttributeDescription(std::string const &k) const;
    std::shared_ptr<data::DataNode> GetDescriptions() const;

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
    SP_OBJECT_HEAD(Attribute, EngineObject)

   protected:
    template <typename... Args>
    explicit Attribute(Args &&... args) : Attribute() {
        db()->SetValue(std::forward<Args>(args)...);
    };

   public:
    virtual std::type_info const &value_type_info() const = 0;
    virtual int GetIFORM() const = 0;
    virtual int GetDOF() const = 0;
    virtual void SetDOF(int) = 0;

    void Register(AttributeGroup *p = nullptr);
    void Deregister(AttributeGroup *p = nullptr);

    std::shared_ptr<data::DataBlock> GetDataBlock();
    std::shared_ptr<const data::DataBlock> GetDataBlock() const;

    int Push(const std::shared_ptr<data::DataNode> &) override;
    std::shared_ptr<data::DataNode> Pop() override;

    virtual bool isNull() const;
    virtual bool empty() const { return isNull(); };
    virtual void Clear();
};
template <typename V, int IFORM>
struct AttributeT : public Attribute {
    typedef V value_type;
    typedef Array<value_type> array_type;
    std::vector<array_type> m_data_;
    AttributeGroup *m_host_ = nullptr;

    template <typename THost, typename... Args>
    explicit AttributeT(THost *host, Args &&... args) : Attribute(std::forward<Args>(args)...), m_dof_(1) {
        Register(dynamic_cast<engine::AttributeGroup *>(host));
    }
    AttributeT() : Attribute(), m_dof_(1) {}
    ~AttributeT() override = default;

    array_type &operator[](int s) { return m_data_[s]; }
    array_type const &operator[](int s) const { return m_data_[s]; }

    std::type_info const &value_type_info() const override { return typeid(V); };
    int GetIFORM() const override { return IFORM; };
    int GetDOF() const override { return m_dof_; };
    void SetDOF(int d) override { m_dof_ = d; };
    auto &Get() { return m_data_; }
    auto const &Get() const { return m_data_; }

    int PushData(std::shared_ptr<data::DataBlock> const &d) {
        //        auto blk = std::dynamic_pointer_cast<data::DataMultiArray<array_type>>(GetDataBlock());
        //        if (blk != nullptr) {
        //            int count = 0;
        //            traits::foreach (*d, [&](array_type& a, auto&&... idx) {
        //                array_type(*blk->Get(count)).swap(a);
        //                ++count;
        //            });
        //        }
        //        Tag();
        return 0;
    };
    std::shared_ptr<data::DataBlock> PopData() {
        //        auto blk = std::dynamic_pointer_cast<data::DataMultiArray<array_type>>(GetDataBlock());
        //        if (blk == nullptr) {
        //            Push(data::DataMultiArray<array_type>::New(d->size()));
        //            blk = std::dynamic_pointer_cast<data::DataMultiArray<array_type>>(GetDataBlock());
        //        }
        //        int count = 0;
        //        traits::foreach (*d, [&](array_type& a, auto&&... idx) {
        //            array_type(a).swap(*blk->Get(count));
        //            a.reset();
        //            ++count;
        //        });
        //        base_type::ResetTag();
        return nullptr;
    };

   private:
    int m_dof_ = 1;
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
