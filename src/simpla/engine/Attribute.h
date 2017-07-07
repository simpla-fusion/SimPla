//
// Created by salmon on 16-10-20.
//

#ifndef SIMPLA_ATTRIBUTEVIEW_H
#define SIMPLA_ATTRIBUTEVIEW_H

#include "MeshBlock.h"
#include "SPObject.h"
#include "simpla/SIMPLA_config.h"
#include "simpla/concept/CheckConcept.h"
#include "simpla/data/all.h"
#include "simpla/utilities/Signal.h"

namespace simpla {
namespace engine {
class Domain;
class Attribute;
class Patch;
class MeshBase;
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

struct AttributeDesc : public data::Configurable {
   public:
    AttributeDesc() = default;
    AttributeDesc(AttributeDesc const &);
    AttributeDesc(AttributeDesc &&other);
    AttributeDesc(int IFORM, int DOF, std::type_info const &t_info, std::string const &s_prefix = "",
                  std::shared_ptr<data::DataTable> const &t_db = nullptr);
    virtual ~AttributeDesc() = default;

    virtual std::string GetPrefix() const;
    virtual int GetIFORM() const;
    virtual int GetDOF() const;
    virtual std::type_info const &value_type_info() const;
    virtual id_type GetID() const;
    virtual std::shared_ptr<AttributeDesc> GetDescription() const;

   private:
    std::string m_prefix_ = "";
    int m_iform_ = 0;
    int m_dof_ = 1;
    std::type_info const &m_t_info_ = typeid(void);
};

class AttributeGroup {
   public:
    AttributeGroup();
    virtual ~AttributeGroup();

    AttributeGroup(AttributeGroup const &other) = delete;
    AttributeGroup(AttributeGroup &&other) = delete;
    AttributeGroup &operator=(AttributeGroup const &other) = delete;
    AttributeGroup &operator=(AttributeGroup &&other) = delete;

    virtual void RegisterDescription(std::map<std::string, std::shared_ptr<AttributeDesc>> *);
    virtual void RegisterAt(AttributeGroup *);
    virtual void DeregisterFrom(AttributeGroup *);

    void Detach(Attribute *attr);
    void Attach(Attribute *attr);

    bool has(std::string const &k) const;
    bool check(std::string const &k, std::type_info const &t_info) const;

    Attribute *Get(std::string const &k);
    Attribute const *Get(std::string const &k) const;

    std::map<std::string, Attribute *> &GetAllAttributes();
    std::map<std::string, Attribute *> const &GetAll() const;

    virtual std::string GetDomainPrefix() const { return ""; }
    virtual MeshBase *GetMesh() = 0;
    virtual MeshBase const *GetMesh() const = 0;

    virtual void Push(Patch *);
    virtual void Pull(Patch *);

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
struct Attribute : public SPObject, public AttributeDesc {
    SP_OBJECT_HEAD(Attribute, SPObject);

   public:
    Attribute(int IFORM, int DOF, std::type_info const &t_info, AttributeGroup *grp,
              std::shared_ptr<data::DataTable> p);
    //    template <int... DOF, typename TGrp>
    //    Attribute(int IFORM, std::integer_sequence<int, DOF...>, std::type_info const &t_info, TGrp *grp,
    //              std::shared_ptr<data::DataTable> cfg)
    //        : Attribute(IFORM, 1, t_info, (grp), cfg) {}
    template <int... DOF, typename TGrp, typename... Args>
    Attribute(int IFORM, std::integer_sequence<int, DOF...>, std::type_info const &t_info, TGrp *grp, Args &&... args)
        : Attribute(IFORM, traits::nProduct<int, DOF...>::value, t_info, (grp),
                    std::make_shared<data::DataTable>(std::forward<Args>(args)...)) {}

    Attribute(Attribute const &other);
    Attribute(Attribute &&other) noexcept;
    ~Attribute() override;

    virtual size_type size() const { return 0; }
    void swap(Attribute &) {}
    void DoUpdate() override;

    const MeshBase *GetMesh() const;

    void Register(AttributeGroup *);

    void Deregister(AttributeGroup *);

    virtual void Push(std::shared_ptr<data::DataBlock>);
    virtual std::shared_ptr<data::DataBlock> Pop();

    virtual bool isNull() const;
    virtual bool empty() const { return isNull(); };

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
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
//    virtual void SetMesh(MeshBase const *){};
//    virtual MeshBase const *GetMesh() const { return nullptr; };
//    virtual void ConvertPatchFromSAMRAI(std::shared_ptr<MeshBlock> const &m, std::shared_ptr<data::DataTable> const
//    &d) {
//        data::data_cast<U>(*d).swap(*this);
//    };
//    virtual std::pair<std::shared_ptr<MeshBlock>, std::shared_ptr<data::DataTable>> Serialize() {
//        return std::make_pair(std::shared_ptr<MeshBlock>(nullptr), data::make_data_entity(*this));
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
