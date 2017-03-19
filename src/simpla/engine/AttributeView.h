//
// Created by salmon on 16-10-20.
//

#ifndef SIMPLA_ATTRIBUTEVIEW_H
#define SIMPLA_ATTRIBUTEVIEW_H

#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/all.h>
#include <simpla/data/all.h>

#include <simpla/concept/CheckConcept.h>
#include <simpla/concept/Printable.h>
#include <simpla/design_pattern/Observer.h>
#include <simpla/design_pattern/Signal.h>

#include "SPObject.h"

namespace simpla {
namespace engine {
class DomainView;
class MeshView;
class MeshBlock;
class AttributeView;
class Patch;
/**
 *  permissions
 *
 *   r : readable
 *   w : writable
 *   c : create/delete
 *
 * [ 0:false 1:true ]
 * 0b0 0 0 0 0
 *   | | | | |------: is shared between different domain
 *   | | | |--------: has ghost cell
 *   | | |----------: PERSISTENT, if false then destroy data when AttributeView is destructed
 *   | |------------: become unmodifiable after first write
 *   |--------------: is coordinate
 */
enum AttributeTag {
    SCRATCH = 0,
    SHARED = 1,            //
    GHOSTED = 1 << 1,      //
    PERSISTENT = 1 << 2,   //
    INPUT = 1 << 3,        //  can only be written once
    COORDINATES = 1 << 4,  //  coordinate of mesh vertex
    NO_FILL,
    GLOBAL = SHARED | GHOSTED | PERSISTENT,
    PRIVATE = GHOSTED | PERSISTENT,
    DEFAULT_ATTRIBUTE_TAG = GLOBAL
};
inline std::ostream &operator<<(std::ostream &os, AttributeTag const &tag) {
    os << std::bitset<32>(static_cast<unsigned long long int>(tag));
    return os;
}
// enum AttributeState { READ = 0b01, WRITE = 0b10 };
//
// struct AttributeDesc : public std::enable_shared_from_this<AttributeDesc> {
//    AttributeDesc(const std::string &name_s, const std::type_info &t_id, int IFORM, int DOF, int TAG = SCRATCH);
//
//    template <typename... Args>
//    AttributeDesc(const std::string &name_s, const std::type_info &t_id, int IFORM, int DOF, int TAG, Args &&... args)
//        : AttributeDesc(name_s, t_id, IFORM, DOF, TAG) {
//        db().Set(std::forward<Args>(args)...);
//    };
//    ~AttributeDesc();
//
//    static id_type GenerateGUID(std::string const &s, std::type_info const &t_id, int IFORM, int DOF,
//                                int tag = SCRATCH);
//
//    std::string const &GetName() const { return m_name_; }
//    const std::type_info &GetValueTypeInfo() const { return m_value_type_info_; }
//    int GetIFORM() const { return m_iform_; }
//    int GetDOF() const { return m_dof_; }
//    int GetTag() const { return m_tag_; }
//    id_type GetGUID() const { return m_GUID_; }
//    data::DataTable &db() { return m_db_; }
//    data::DataTable const &db() const { return m_db_; }
//
//   private:
//    const std::string m_name_;
//    const std::type_info &m_value_type_info_;
//    int m_iform_;
//    int m_dof_;
//    int m_tag_;
//    id_type m_GUID_;
//    data::DataTable m_db_;
//};

class AttributeViewBundle {
   public:
    AttributeViewBundle(DomainView *p = nullptr);
    virtual ~AttributeViewBundle();
    DomainView *GetDomain() const;
    void Detach(AttributeView *attr);
    void Attach(AttributeView *attr);
    virtual MeshView const *GetMesh() const;
    virtual void SetPatch(std::shared_ptr<Patch> const &);
    virtual std::shared_ptr<Patch> GetPatch() const;
    void Foreach(std::function<void(AttributeView *)> const &) const;

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
struct AttributeView : public SPObject {
    SP_OBJECT_BASE(AttributeView);

   public:
    AttributeView(AttributeViewBundle *b = nullptr, std::shared_ptr<data::DataEntity> const &p = nullptr);
    AttributeView(AttributeView const &other) = delete;
    AttributeView(AttributeView &&other) = delete;
    virtual ~AttributeView();

   public:
    void Config();
    void Config(std::string const &s) { db()->SetValue("name", s); }
    void Config(AttributeTag const &s) { db()->SetValue("tag", (s)); }
    void Config(data::KeyValue const &s) { db()->SetValue(s); }
    void Config(int const &s) { db()->SetValue("tag", (s)); }

    template <typename T0, typename T1, typename... Others>
    void Config(T0 const &a0, T1 const &a1, Others &&... others) {
        Config(a0);
        Config(a1, std::forward<Others>(others)...);
    }

    id_type GetGUID() const;
    int GetTag() const;
    virtual int GetIFORM() const = 0;
    virtual int GetDOF() const = 0;
    virtual std::type_info const &value_type_info() const = 0;  //!< value type
    virtual std::type_info const &mesh_type_info() const = 0;   //!< mesh type

    virtual void SetData(std::shared_ptr<data::DataEntity> const &);
    virtual std::shared_ptr<data::DataEntity> GetData() const;
    virtual bool Update();
    bool isNull() const;
    bool empty() const { return isNull(); };

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

template <typename, typename Enable = void>
class AttributeViewAdapter {};

CHECK_TYPE_MEMBER(mesh_type, mesh_type);

template <typename U>
class AttributeViewAdapter<U, std::enable_if_t<!has_mesh_type<U>::value>> : public AttributeView, public U {
    SP_OBJECT_HEAD(AttributeViewAdapter<U>, AttributeView);

    typedef algebra::traits::value_type_t<U> value_type;
    static const int iform = algebra::traits::iform<U>::value;
    static const int dof = algebra::traits::dof<U>::value;

   public:
    typedef std::true_type prefer_pass_by_reference;

    template <typename... Args>
    explicit AttributeViewAdapter(AttributeViewBundle *b, Args &&... args) : AttributeView(b) {
        Config(std::forward<Args>(args)...);
    }

    AttributeViewAdapter(AttributeViewAdapter &&) = delete;
    AttributeViewAdapter(AttributeViewAdapter const &) = delete;
    virtual ~AttributeViewAdapter() {}

    //    virtual std::shared_ptr<AttributeView> Duplicate(std::string const &s = "", int TAG = NORMAL) const = 0;
    virtual std::type_info const &value_type_info() const { return typeid(value_type); };  //!< value type
    virtual int GetIFORM() const { return iform; };
    virtual int GetDOF() const { return dof; };
    void InitializeData() {}
    virtual void Clear() { U::Clear(); }

    template <typename TExpr>
    this_type &operator=(TExpr const &expr) {
        Click();
        U::operator=(expr);
        return *this;
    };

    bool Update() final {
        if (AttributeView::Update()) {
            U::Update();
            return false;
        } else {
            return true;
        }
    }
};
template <typename U>
class AttributeViewAdapter<U, std::enable_if_t<has_mesh_type<U>::value>> : public AttributeView, public U {
    typedef AttributeViewAdapter<U, std::enable_if_t<has_mesh_type<U>::value>> attribute_type;
    SP_OBJECT_HEAD(attribute_type, AttributeView);

    typedef typename U::mesh_type mesh_type;
    typedef algebra::traits::value_type_t<U> value_type;
    static const int iform = algebra::traits::iform<U>::value;
    static const int dof = algebra::traits::dof<U>::value;
    static const int NDIMS = algebra::traits::ndims<U>::value;
    static const int num_of_sub = algebra::traits::num_of_sub<U>::value;

   public:
    typedef std::true_type prefer_pass_by_reference;
    typedef Array<value_type, NDIMS> array_type;
    template <typename... Args>
    explicit AttributeViewAdapter(AttributeViewBundle *b, Args &&... args)
        : AttributeView(b), U(static_cast<mesh_type const *>(b->GetMesh())) {
        Config(std::forward<Args>(args)...);
    }

    AttributeViewAdapter(AttributeViewAdapter &&) = delete;
    AttributeViewAdapter(AttributeViewAdapter const &) = delete;
    virtual ~AttributeViewAdapter() {}

    virtual std::type_info const &mesh_type_info() const { return typeid(mesh_type); };    //!< mesh type
    virtual std::type_info const &value_type_info() const { return typeid(value_type); };  //!< value type
    virtual int GetIFORM() const { return iform; };
    virtual int GetDOF() const { return dof; };
    //    virtual mesh_type const *GetMesh() const { return static_cast<mesh_type const *>(GetDomain()->GetMesh()); }
    virtual void Clear() { U::Clear(); }

    template <typename TExpr>
    this_type &operator=(TExpr const &expr) {
        Click();
        U::operator=(expr);
        return *this;
    };
    virtual void SetData(std::shared_ptr<data::DataEntity> const &d) {
        AttributeView::SetData(d);
        if (GetDOF() == 1 && d->isHeavyBlock()) {
            auto p = d->template cast_as<data::DataEntityWrapper<array_type>>().data();
            U::SetData(&p);
        } else if (d->isArray() && d->cast_as<data::DataArray>().size() == num_of_sub) {
            std::vector<std::shared_ptr<array_type>> t_d;
            d->cast_as<data::DataArray>().Foreach([&](std::shared_ptr<data::DataEntity> const &v) {
                t_d.push_back(v->template cast_as<data::DataEntityWrapper<array_type>>().data());
            });
            U::SetData(&t_d[0]);
        }
    }
    virtual std::shared_ptr<data::DataEntity> GetData() const {}
    bool Update() final {
        if (AttributeView::Update()) {
            U::Update();
            return false;
        } else {
            return true;
        }
    }
};

template <typename TV, typename TM, int IFORM = VERTEX, int DOF = 1>
using FieldAttribute = AttributeViewAdapter<Field<TV, TM, IFORM, DOF>>;

template <typename TV = Real, int IFORM = VERTEX, int DOF = 1>
using DataAttribute = AttributeViewAdapter<Array<TV, 3 + (((IFORM == VERTEX || IFORM == VOLUME) && DOF == 1) ? 0 : 1)>>;
//
// template <typename TV, int IFORM = VERTEX, int DOF = 1>
// struct DataAttribute : public AttributeView,
//                       public Array<TV, 3 + (((IFORM == VERTEX || IFORM == VOLUME) && DOF == 1) ? 0 : 1)> {
//    typedef Array<TV, 3 + (((IFORM == VERTEX || IFORM == VOLUME) && DOF == 1) ? 0 : 1)> array_type;
//    typedef DataAttribute<TV, IFORM, DOF> data_attr_type;
//    SP_OBJECT_HEAD(data_attr_type, AttributeView);
//    CHOICE_TYPE_WITH_TYPE_MEMBER(mesh_traits, mesh_type, MeshView)
//    typedef TV value_type;
//    static constexpr int GetIFORM = IFORM;
//    static constexpr int GetDOF = DOF;
//    typedef MeshView mesh_type;
//
//    template <typename TM, typename... Args>
//    DataAttribute(TM *w, Args &&... args)
//        : base_type(w, AttributeDesc::create<value_type, GetIFORM, GetDOF>(std::forward<Args>(args)...)),
//          AttributeView(<#initializer #>, nullptr, <#initializer #>) {}
//    template <typename TM>
//    DataAttribute(TM *m, std::initializer_list<data::KeyValue> const &param)
//        : base_type(m, AttributeDesc::create<value_type, GetIFORM, GetDOF>(param)),
//          AttributeView(<#initializer #>, nullptr, <#initializer #>) {}
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
//    static std::shared_ptr<this_type> make_shared(MeshView *c, std::initializer_list<data::KeyValue> const &param) {
//        return std::make_shared<this_type>(c, param);
//    }
//    virtual std::ostream &Print(std::ostream &os, int indent = 0) const { return array_type::Print(os, indent); }
//
//    virtual value_type *data() { return reinterpret_cast<value_type *>(AttributeView::GetDataBlock()->raw_data()); }
//
//    virtual void Update() {
//        AttributeView::Update();
//        array_type::Update();
//    }
//    virtual void Finalize() {
//        array_type::Finalize();
//        AttributeView::Finalize();
//    }
//
//    virtual void Clear() { array_type::Clear(); }
//};

}  // namespace engine
}  // namespace simpla

#endif  // SIMPLA_ATTRIBUTEVIEW_H
