//
// Created by salmon on 16-10-20.
//

#ifndef SIMPLA_ATTRIBUTEVIEW_H
#define SIMPLA_ATTRIBUTEVIEW_H

#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/Array.h>
#include <simpla/algebra/all.h>
#include <simpla/concept/Configurable.h>
#include <simpla/concept/Printable.h>
#include <simpla/concept/Serializable.h>
#include "AttributeDesc.h"
#include "Object.h"
namespace simpla {

namespace engine {
class DomainView;
class MeshView;
class DataBlock;
class AttributeDesc;
/**
 * @startuml
 * title Life cycle
 * actor Main
 *  participant AttributeView
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
struct AttributeView : public concept::Printable {
   public:
    SP_OBJECT_BASE(AttributeView);

    AttributeView(DomainView *w = nullptr, AttributeDesc const *desc = nullptr);
    AttributeView(AttributeView const &other) = delete;
    AttributeView(AttributeView &&other) = delete;
    virtual ~AttributeView();

    virtual std::ostream &Print(std::ostream &os, int indent = 0) const { return os; };
    virtual void Initialize();
    virtual AttributeDesc const &description() const;

    const std::shared_ptr<DataBlock> &data_block() const;
    std::shared_ptr<DataBlock> &data_block();
    MeshView const *mesh_view() const;
    bool isUpdated() const;
    void Update();
    bool isNull() const;
    bool empty() const { return isNull(); };

    void SetDomain(DomainView *d);
    DomainView const *GetDomain() const;
    void UnsetDomain();

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
    static constexpr int iform = algebra::traits::iform<U>::value;
    static constexpr int dof = algebra::traits::dof<U>::value;
    typedef mesh_traits_t<U> mesh_type;
    std::shared_ptr<AttributeDesc> m_desc_;

   public:
    template <typename... Args>
    AttributeViewAdapter(DomainView *m, Args &&... args)
        : AttributeView(m), m_desc_(AttributeDesc::create<value_type, iform, dof>(std::forward<Args>(args)...)) {}
    AttributeViewAdapter(DomainView *w, std::initializer_list<data::KeyValue> const &param)
        : AttributeView(w), m_desc_(AttributeDesc::create<value_type, iform, dof>(param)) {}

    AttributeViewAdapter(AttributeViewAdapter &&) = delete;
    AttributeViewAdapter(AttributeViewAdapter const &) = delete;
    virtual ~AttributeViewAdapter() {}
    virtual AttributeDesc const &description() const { return *m_desc_; }

    virtual std::shared_ptr<DataBlock> CreateDataBlock() const {
        std::shared_ptr<DataBlock> p = AttributeView::data_block();

        if (p == nullptr) {
            UNIMPLEMENTED;
            std::shared_ptr<DataBlock> d(nullptr);
            //        if (d == nullptr) {
            //            return std::make_shared<DefaultDataBlock<value_type, iform, dof>>(nullptr, U::size());
            //        } else {
            //            return std::make_shared<DefaultDataBlock<value_type, iform, dof>>(
            //                std::shared_ptr<value_type>(static_cast<value_type *>(d), simpla::tags::do_nothing()),
            //                U::size());
            //        }
        }
        return p;
    };

    using U::operator=;
    virtual void Update() {
        AttributeView::Update();
        U::Update();
    }
    template <typename TM>
    static std::shared_ptr<this_type> make_shared(TM *c, std::initializer_list<data::KeyValue> const &param) {
        return std::make_shared<this_type>(c, param);
    }
    virtual std::ostream &Print(std::ostream &os, int indent = 0) const { return U::print(os, indent); }

    virtual mesh_type const *mesh() const {
        static_assert(std::is_base_of<mesh::MeshView, mesh_type>::value, "mesh is  not base on MeshVew");
        return static_cast<mesh_type const *>(mesh_view());
    };
    virtual value_type *data() { return reinterpret_cast<value_type *>(data_block()->data()); }
    virtual value_type const *data() const { return reinterpret_cast<value_type *>(data_block()->data()); }
};

template <typename TV, typename TM, int IFORM = VERTEX, int DOF = 1>
using FieldAttribute = AttributeViewAdapter<Field<TV, TM, IFORM, DOF>>;

template <typename TV, int IFORM = VERTEX, int DOF = 1>
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
//    virtual std::shared_ptr<DataBlock> CreateDataBlock(void *p = nullptr) const {
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
//    virtual value_type *data() { return reinterpret_cast<value_type *>(AttributeView::data_block()->raw_data()); }
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
