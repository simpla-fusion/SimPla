//
// Created by salmon on 16-10-20.
//

#ifndef SIMPLA_ATTRIBUTEVIEW_H
#define SIMPLA_ATTRIBUTEVIEW_H

#include "MeshBlock.h"
#include "SPObject.h"
#include "simpla/SIMPLA_config.h"
#include "simpla/concept/CheckConcept.h"
#include "simpla/data/Serializable.h"
#include "simpla/data/all.h"
#include "simpla/design_pattern/Signal.h"
namespace simpla {
namespace engine {
class Domain;
class Mesh;
class MeshBlock;
class Attribute;
class Patch;
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

template <typename TV = double, int IFORM = VERTEX, int DOF = 1>
struct AttributeDesc;
class AttributeGroup {
   public:
    AttributeGroup();
    virtual ~AttributeGroup();

    virtual void Register(AttributeGroup *);
    virtual void Deregister(AttributeGroup *);

    void Detach(Attribute *attr);
    void Attach(Attribute *attr);

    //    virtual void Push(const std::shared_ptr<Patch> &);
    //    virtual std::shared_ptr<Patch> Pop();

    std::set<Attribute *> const &GetAll() const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

/**
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
 */
struct Attribute : public SPObject, public concept::Configurable, public data::Serializable {
    SP_OBJECT_BASE(Attribute);

   public:
    Attribute(std::shared_ptr<data::DataTable> const &t = nullptr);

    Attribute(AttributeGroup *b, std::shared_ptr<data::DataTable> const &p);
    Attribute(AttributeGroup *b) : Attribute(b, std::shared_ptr<data::DataTable>(nullptr)){};
    template <typename U, typename... Args>
    explicit Attribute(AttributeGroup *b, U const &first, Args &&... args)
        : Attribute(b, std::make_shared<data::DataTable>(first, std::forward<Args>(args)...)){};
    Attribute(Attribute const &other) = delete;
    Attribute(Attribute &&other) = delete;
    virtual ~Attribute();

    void Register(AttributeGroup *);
    void Deregister(AttributeGroup *);
    //    virtual Attribute *Clone() const = 0;
    //    virtual std::shared_ptr<Attribute> GetDescription() const = 0;
    virtual int GetIFORM() const = 0;
    virtual int GetDOF() const = 0;
    virtual std::type_info const &value_type_info() const = 0;  //!< value type

    void SetMesh(Mesh const *);
    Mesh const *GetMesh() const;

    virtual void PushData(std::shared_ptr<data::DataBlock> const &){};
    virtual std::shared_ptr<data::DataBlock> PopData() { return nullptr; }

    virtual bool isNull() const;
    virtual bool empty() const { return isNull(); };

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
template <typename TV, int IFORM, int DOF>
struct AttributeDesc : public Attribute {
    typedef AttributeDesc<TV, IFORM, DOF> desc_type;
    SP_OBJECT_HEAD(desc_type, Attribute);

   public:
    AttributeDesc(std::shared_ptr<data::DataTable> const &t) : Attribute(t) {}
    AttributeDesc(std::string const &k) : Attribute() {}
    ~AttributeDesc() {}

    virtual std::shared_ptr<Attribute> GetDescription() const {
        return std::make_shared<AttributeDesc<TV, IFORM, DOF>>();
    };
    //    virtual Attribute *Clone() const { return new this_type; };

    virtual int GetIFORM() const { return IFORM; };
    virtual int GetDOF() const { return DOF; };
    virtual std::type_info const &value_type_info() const { return typeid(TV); };
};

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
//    SP_OBJECT_HEAD(AttributeViewAdapter<U>, Attribute);
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
//        : Attribute(b, data::make_data_entity(std::forward<Args>(args)...)) {}
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
//    virtual void SetMesh(Mesh const *){};
//    virtual Mesh const *GetMesh() const { return nullptr; };
//    virtual void Push(std::shared_ptr<MeshBlock> const &m, std::shared_ptr<data::DataTable> const &d) {
//        data::data_cast<U>(*d).swap(*this);
//    };
//    virtual std::pair<std::shared_ptr<MeshBlock>, std::shared_ptr<data::DataTable>> PopPatch() {
//        return std::make_pair(std::shared_ptr<MeshBlock>(nullptr), data::make_data_entity(*this));
//    };
//    template <typename TExpr>
//    this_type &operator=(TExpr const &expr) {
//        Click();
//        U::operator=(expr);
//        return *this;
//    };
//
//    bool Update() final {
//        if (!Attribute::Update()) { return false; }
//        return U::Update();
//    }
//};
//
// template <typename TV, typename TM, int IFORM = VERTEX, int DOF = 1>
// using FieldAttribute = Field<TV, TM, IFORM, DOF>;
//
// template <typename TV = Real, int IFORM = VERTEX, int DOF = 1>
// using DataAttribute = AttributeViewAdapter<Array<TV, 3 + (((IFORM == VERTEX || IFORM == VOLUME) && DOF == 1) ? 0 :
// 1)>>;
//
// template <typename TV, int IFORM = VERTEX, int DOF = 1>
// struct DataAttribute : public Attribute,
//                       public Array<TV, 3 + (((IFORM == VERTEX || IFORM == VOLUME) && DOF == 1) ? 0 : 1)> {
//    typedef Array<TV, 3 + (((IFORM == VERTEX || IFORM == VOLUME) && DOF == 1) ? 0 : 1)> array_type;
//    typedef DataAttribute<TV, IFORM, DOF> data_attr_type;
//    SP_OBJECT_HEAD(data_attr_type, Attribute);
//    CHOICE_TYPE_WITH_TYPE_MEMBER(mesh_traits, mesh_type, Mesh)
//    typedef TV value_type;
//    static constexpr int GetIFORM = IFORM;
//    static constexpr int GetDOF = DOF;
//    typedef Mesh mesh_type;
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
//    virtual std::shared_ptr<DataBlock> InitializeData(void *p = nullptr) const {
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
//    static std::shared_ptr<this_type> make_shared(Mesh *c, std::initializer_list<data::KeyValue> const &param) {
//        return std::make_shared<this_type>(c, param);
//    }
//    virtual std::ostream &Print(std::ostream &os, int indent = 0) const { return array_type::Print(os, indent); }
//
//    virtual value_type *data() { return reinterpret_cast<value_type *>(Attribute::GetDataBlock()->raw_data()); }
//
//    virtual void Update() {
//        Attribute::Update();
//        array_type::Update();
//    }
//    virtual void Finalize() {
//        array_type::Finalize();
//        Attribute::Finalize();
//    }
//
//    virtual void Clear() { array_type::Clear(); }
//};

}  // namespace engine
}  // namespace simpla

#endif  // SIMPLA_ATTRIBUTEVIEW_H
