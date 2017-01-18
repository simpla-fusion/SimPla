//
// Created by salmon on 16-10-20.
//

#ifndef SIMPLA_ATTRIBUTE_H
#define SIMPLA_ATTRIBUTE_H

#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/Array.h>
#include <simpla/algebra/all.h>
#include <simpla/concept/Configurable.h>
#include <simpla/concept/Object.h>
#include <simpla/concept/Printable.h>
#include <simpla/concept/Serializable.h>
#include <simpla/data/all.h>
#include "DataBlock.h"

namespace simpla {
namespace mesh {
class Mesh;

struct AttributeDesc : public Object, public concept::Configurable, public concept::Printable {
    AttributeDesc() : m_value_type_index_(std::type_index(typeid(Real))), m_iform_(VERTEX), m_dof_(1) { deploy(); }

    AttributeDesc(AttributeDesc const &other) = delete;

    AttributeDesc(AttributeDesc &&other) = delete;

    virtual ~AttributeDesc() {}

    virtual void deploy() {
        m_value_type_index_ = value_type_index();
        m_iform_ = entity_type();
        m_dof_ = dof();
    }

    template <typename ValueType, int IFORM, int DOF>
    static std::shared_ptr<AttributeDesc> create() {
        auto res = std::make_shared<AttributeDesc>();
        res->m_iform_ = IFORM;
        res->m_dof_ = DOF;
        res->m_value_type_index_ = std::type_index(typeid(ValueType));
        return res;
    };
    template <typename ValueType, int IFORM, int DOF>
    static std::shared_ptr<AttributeDesc> create(std::initializer_list<data::KeyValue> const &param) {
        auto res = create<ValueType, IFORM, DOF>();
        res->db.insert(param);
        return res;
    };
    virtual std::type_index value_type_index() const { return m_value_type_index_; };

    virtual int entity_type() const { return m_iform_; };

    virtual int dof() const { return m_dof_; };
    template <typename U>
    bool check_value_type() const {
        return std::type_index(typeid(U)) == m_value_type_index_;
    }

   private:
    std::type_index m_value_type_index_;
    int m_iform_ = VERTEX;
    int m_dof_ = 1;
};

class AttributeDict : public concept::Printable {
   public:
    virtual std::ostream &print(std::ostream &os, int indent = 0) const;

    std::pair<std::shared_ptr<AttributeDesc>, bool> register_attr(std::shared_ptr<AttributeDesc> const &desc);

    void erase(id_type const &id);

    void erase(std::string const &id);

    std::shared_ptr<AttributeDesc> find(id_type const &id);

    std::shared_ptr<AttributeDesc> find(std::string const &id);

    std::shared_ptr<AttributeDesc> const &get(std::string const &k) const;

    std::shared_ptr<AttributeDesc> const &get(id_type k) const;

   private:
    std::map<std::string, id_type> m_key_id_;
    std::map<id_type, std::shared_ptr<AttributeDesc>> m_map_;
};

struct Attribute : public concept::Printable, public concept::LifeControllable {
   public:
    SP_OBJECT_BASE(Attribute);

    Attribute(Mesh *m, const std::shared_ptr<AttributeDesc> &desc, const std::shared_ptr<DataBlock> &d = nullptr);

    Attribute(Attribute const &other) = delete;

    Attribute(Attribute &&other) = delete;

    virtual ~Attribute();

    virtual std::ostream &print(std::ostream &os, int indent = 0) const { return os; };

    Mesh *mesh() { return m_mesh_; }

    Mesh const *mesh() const { return m_mesh_; }

    AttributeDesc const &description() const { return *m_desc_; }

    std::shared_ptr<DataBlock> const &data_block() const { return m_data_; }

    std::shared_ptr<DataBlock> &data_block() { return m_data_; }

    void accept(std::shared_ptr<DataBlock> const &d);

    virtual void pre_process();

    virtual void post_process();

    virtual void clear();

   private:
    Mesh *m_mesh_;
    std::shared_ptr<AttributeDesc> m_desc_ = nullptr;
    std::shared_ptr<DataBlock> m_data_;
};

template <typename...>
class AttributeAdapter;

template <typename U>
class AttributeAdapter<U> : public Attribute, public U {
    SP_OBJECT_HEAD(AttributeAdapter<U>, Attribute);
    CHOICE_TYPE_WITH_TYPE_MEMBER(mesh_traits, mesh_type, Mesh)
    typedef algebra::traits::value_type_t<U> value_type;
    static constexpr int iform = algebra::traits::iform<U>::value;
    static constexpr int dof = algebra::traits::dof<U>::value;
    typedef mesh_traits_t<U> mesh_type;

   public:
    template <typename... Args>
    AttributeAdapter(Mesh *m, Args &&... args)
        : base_type(m, AttributeDesc::create<value_type, iform, dof>(std::forward<Args>(args)...)) {}

    AttributeAdapter(Mesh *m, std::initializer_list<data::KeyValue> const &param)
        : base_type(m, AttributeDesc::create<value_type, iform, dof>(param)) {}

    AttributeAdapter(AttributeAdapter &&) = delete;

    AttributeAdapter(AttributeAdapter const &) = delete;

    virtual ~AttributeAdapter() {}

    virtual std::shared_ptr<DataBlock> create_data_block(void *p = nullptr) const {
        UNIMPLEMENTED;
        return std::shared_ptr<DataBlock>(nullptr);
    };

    using U::operator=;
    template <typename... Args>
    static std::shared_ptr<this_type> make_shared(Args &&... args) {
        return std::make_shared<this_type>(std::forward<Args>(args)...);
    }

    static std::shared_ptr<this_type> make_shared(Mesh *c, std::initializer_list<data::KeyValue> const &param) {
        return std::make_shared<this_type>(c, param);
    }
    virtual std::ostream &print(std::ostream &os, int indent = 0) const { return U::print(os, indent); }

    virtual mesh_type *mesh() { return static_cast<mesh_type *>(Attribute::mesh()); };

    virtual mesh_type const *mesh() const { return static_cast<mesh_type const *>(Attribute::mesh()); };

    virtual std::shared_ptr<value_type> data() {
        return std::shared_ptr<value_type>(reinterpret_cast<value_type *>(Attribute::data_block()->data()),
                                           simpla::tags::do_nothing());
    }

    virtual void deploy() {
        Attribute::deploy();
        U::deploy();
    }

    template <typename... Args>
    static this_type create(Args &&... args) {
        std::make_shared<this_type>(std::forward<Args>(args)...);
    }

    //    virtual std::shared_ptr<DataBlock> create_data_block(MeshBlock const *m, void *p = nullptr) const {
    //        return DataBlockAdapter<value_type>::create(m, static_cast<value_type *>(p));
    //    };

    virtual void clear() { U::clear(); }

    virtual void pre_process() { Attribute::pre_process(); };

    virtual void post_process() { Attribute::post_process(); }
};

template <typename TV, int IFORM = VERTEX, int DOF = 1>
using DataAttribute = AttributeAdapter<Array<TV, 3 + (((IFORM == VERTEX || IFORM == VOLUME) && DOF == 1) ? 0 : 1)>>;
template <typename TV, typename TM, int IFORM = VERTEX, int DOF = 1>
using FieldAttribute = AttributeAdapter<Field<TV, TM, IFORM, DOF>>;
}
}  // namespace data_block

#endif  // SIMPLA_ATTRIBUTE_H
