//
// Created by salmon on 16-10-20.
//

#ifndef SIMPLA_ATTRIBUTEVIEW_H
#define SIMPLA_ATTRIBUTEVIEW_H

#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/all.h>
#include <simpla/concept/Printable.h>
#include <simpla/concept/StateCounter.h>
#include <simpla/design_pattern/Observer.h>
#include <simpla/design_pattern/Signal.h>

#include "SPObject.h"

namespace simpla {
namespace engine {
class DomainView;
class MeshView;
class DataBlock;
class AttributeView;
enum AttributeTag { NORMAL = 0, SCRATCH = 0b100, INPUT = 0b1000, COORDINATES, NO_FILL, CHECK = 0x0100 };
enum AttributeLockState { READ = 0b01, WRITE = 0b10 };

struct AttributeDesc {
    AttributeDesc(const std::string &name_s, const std::type_info &t_id, int IFORM, int DOF, AttributeTag TAG);

    template <typename... Args>
    AttributeDesc(const std::string &name_s, const std::type_info &t_id, int IFORM, int DOF, AttributeTag TAG,
                  Args &&... args)
        : AttributeDesc(name_s, t_id, IFORM, DOF, TAG) {
        db().SetValue(std::forward<Args>(args)...);
    };
    ~AttributeDesc();

    static id_type GenerateGUID(std::string const &s, std::type_info const &t_id, int IFORM, int DOF, AttributeTag tag);

    std::string const &name() const { return m_name_; }
    const std::type_info &value_type_info() const { return m_value_type_info_; }
    int iform() const { return m_iform_; }
    int dof() const { return m_dof_; }
    AttributeTag tag() const { return m_tag_; }
    id_type GUID() const { return m_GUID_; }
    data::DataTable &db() { return m_db_; }
    data::DataTable const &db() const { return m_db_; }

   private:
    const std::string m_name_;
    const std::type_info &m_value_type_info_;
    int m_iform_;
    int m_dof_;
    AttributeTag m_tag_;
    id_type m_GUID_;
    data::DataTable m_db_;
};
struct AttributeDict : public concept::Printable {
   public:
    AttributeDict();
    virtual ~AttributeDict();
    virtual std::ostream &Print(std::ostream &os, int indent = 0) const;

    bool has(id_type) const;
    bool has(std::string const &) const;
    id_type GetGUID(std::string const &) const;
    std::shared_ptr<AttributeDesc> Get(id_type) const;
    std::shared_ptr<AttributeDesc> Get(std::string const &) const;
    std::shared_ptr<AttributeDesc> Set(id_type gid, std::shared_ptr<AttributeDesc>);
    void Remove(id_type);
    void Remove(const std::string &);
    void for_each(std::function<void(AttributeDesc *)> const &);
    void for_each(std::function<void(AttributeDesc const *)> const &) const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
class AttributeViewBundle : public SPObject, public concept::Printable {
   public:
    AttributeViewBundle();
    virtual ~AttributeViewBundle();
    virtual std::ostream &Print(std::ostream &os, int indent) const;

    virtual bool isModified();
    virtual bool Update();

    DomainView const &GetDomain() const;
    MeshView const &GetMesh() const;
    std::shared_ptr<DataBlock> &GetDataBlock(id_type guid) const;

    void Detach(AttributeView *attr);
    void Attach(AttributeView *attr);
    void OnNotify();

    void Accept(std::function<void(AttributeView *)> const &) const;

    void RegisterAttribute(AttributeDict *dbase);

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
struct AttributeView : public SPObject, public concept::Printable {
   public:
    SP_OBJECT_BASE(AttributeView);

   private:
    AttributeView();

   public:
    AttributeView(MeshView const *b);
    AttributeView(AttributeViewBundle *b);
    template <typename T>
    AttributeView(T *b,
                  ENABLE_IF((std::is_base_of<MeshView, T>::value && std::is_base_of<AttributeViewBundle, T>::value)))
        : AttributeView(static_cast<AttributeViewBundle *>(b)){
              //        SetMesh(static_cast<MeshView const *>(b));
          };

    AttributeView(AttributeView const &other) = delete;
    AttributeView(AttributeView &&other) = delete;
    virtual ~AttributeView();

   protected:
    void Config(std::string const &s = "unnamed", AttributeTag t = SCRATCH);

    template <typename... Others>
    void Config(std::string const &s, AttributeTag t, Others &&... others) {
        Config(s, t);
        db().SetValue(std::forward<Others>(others)...);
    };

   public:
    virtual std::ostream &Print(std::ostream &os, int indent = 0) const;

    AttributeDesc &description() const;
    data::DataTable &db() const;
    id_type GUID() const;
    AttributeTag tag() const;
    std::string const &name() const;
    virtual int iform() const;
    virtual int dof() const;
    virtual std::type_info const &value_type_info() const;     //!< value type
    virtual std::type_info const &mesh_type_info() const = 0;  //!< mesh type

    virtual bool Update();

    bool isNull() const;
    bool empty() const { return isNull(); };

    /**
     * @ingroup { observer
     */
    void Connect(AttributeViewBundle *b);
    void OnNotify();
    /** @}*/

    MeshView const &GetMesh() const;
    DataBlock &GetDataBlock() const;

    virtual void InitializeData();

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

template <typename...>
class AttributeViewAdapter;

template <typename U>
class AttributeViewAdapter<U> : public AttributeView, public U {
    SP_OBJECT_HEAD(AttributeViewAdapter<U>, AttributeView);

    CHOICE_TYPE_WITH_TYPE_MEMBER(mesh_traits, mesh_type, MeshView)
    typedef algebra::traits::value_type_t<U> value_type;
    typedef mesh_traits_t<U> mesh_type;

   public:
    typedef std::true_type prefer_pass_by_reference;
    template <typename TB, typename... Args>
    explicit AttributeViewAdapter(TB *b, Args &&... args) : AttributeView(b) {
        AttributeView::Config(std::forward<Args>(args)...);
    }

    AttributeViewAdapter(AttributeViewAdapter &&) = delete;
    AttributeViewAdapter(AttributeViewAdapter const &) = delete;
    virtual ~AttributeViewAdapter() {}

    std::ostream &Print(std::ostream &os, int indent = 0) const final {
        os << AttributeView::description().name() << " = {";
        U::Print(os, indent);
        os << "}";
        return os;
    }
    //    virtual std::shared_ptr<AttributeView> Clone(std::string const &s = "", int TAG = NORMAL) const = 0;
    virtual std::type_info const &mesh_type_info() const { return typeid(mesh_type); };    //!< mesh type
    virtual std::type_info const &value_type_info() const { return typeid(value_type); };  //!< value type
    virtual int iform() const { return algebra::traits::iform<U>::value; };
    virtual int dof() const { return algebra::traits::dof<U>::value; };
    void InitializeData() {}
    std::shared_ptr<DataBlock> CreateDataBlock() const {
        std::shared_ptr<DataBlock> p = nullptr;

        // TODO: create data block!!
        //        if (p == nullptr) {
        //            UNIMPLEMENTED;
        //            std::shared_ptr<DataBlock> d(nullptr);
        //            //        if (d == nullptr) {
        //            //            return std::make_shared<DefaultDataBlock<value_type, iform, dof>>(nullptr,
        //            U::size());
        //            //        } else {
        //            //            return std::make_shared<DefaultDataBlock<value_type, iform, dof>>(
        //            //                std::shared_ptr<value_type>(static_cast<value_type *>(d),
        //                simpla::tags::do_nothing()),
        //                //                U::size());
        //                //        }
        //        }
        return p;
    };
    template <typename TExpr>
    this_type &operator=(TExpr const &expr) {
        Click();
        U::operator=(expr);
        return *this;
    };

    virtual mesh_type const *mesh() const {
        static_assert(std::is_base_of<MeshView, mesh_type>::value, "illegal mesh_type");
        return static_cast<mesh_type const *>(&AttributeView::GetMesh());
    }
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
//    static constexpr int iform = IFORM;
//    static constexpr int dof = DOF;
//    typedef MeshView mesh_type;
//
//    template <typename TM, typename... Args>
//    DataAttribute(TM *w, Args &&... args)
//        : base_type(w, AttributeDesc::create<value_type, iform, dof>(std::forward<Args>(args)...)),
//          AttributeView(<#initializer #>, nullptr, <#initializer #>) {}
//    template <typename TM>
//    DataAttribute(TM *m, std::initializer_list<data::KeyValue> const &param)
//        : base_type(m, AttributeDesc::create<value_type, iform, dof>(param)),
//          AttributeView(<#initializer #>, nullptr, <#initializer #>) {}
//    DataAttribute(DataAttribute &&) = delete;
//    DataAttribute(DataAttribute const &) = delete;
//    virtual ~DataAttribute() {}
//
//    template <typename... Args>
//    static std::shared_ptr<this_type> Create(Args &&... args) {
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
