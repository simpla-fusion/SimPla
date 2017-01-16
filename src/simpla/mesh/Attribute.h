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
#include <simpla/design_pattern/Observer.h>

#include "DataBlock.h"

namespace simpla {
namespace mesh {
class Patch;
class Chart;
class Attribute;
class AttributeCollection;

struct AttributeDesc : public concept::Configurable,
                       public Object,
                       std::enable_shared_from_this<AttributeDesc> {
    AttributeDesc()
        : m_value_type_index_(std::type_index(typeid(Real))), m_iform_(VERTEX), m_dof_(1) {
        deploy();
    }

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

    std::pair<std::shared_ptr<AttributeDesc>, bool> register_attr(
        std::shared_ptr<AttributeDesc> const &desc);

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

struct Attribute : public concept::Printable,
                   public concept::LifeControllable,
                   public design_pattern::Observer<void(Patch *)> {
   public:
    SP_OBJECT_BASE(Attribute);


    Attribute(const std::shared_ptr<AttributeDesc> &desc,
              const std::shared_ptr<DataBlock> &d = nullptr);

    Attribute(std::shared_ptr<AttributeDesc> const &desc, AttributeCollection *p);

    Attribute(Attribute const &other) = delete;

    Attribute(Attribute &&other) = delete;

    virtual ~Attribute();

    virtual std::ostream &print(std::ostream &os, int indent = 0) const { return os; };

    virtual std::shared_ptr<DataBlock> create_data_block(MeshBlock const *m,
                                                         void *p = nullptr) const = 0;

    virtual AttributeDesc &description(std::shared_ptr<AttributeDesc> const &desc = nullptr) {
        if (desc != nullptr) { m_desc_ = desc; }
        return *m_desc_;
    }

    virtual AttributeDesc const &description() const { return *m_desc_; }

    virtual std::shared_ptr<DataBlock> const &data_block() const { return m_data_; }

    virtual std::shared_ptr<DataBlock> &data_block() { return m_data_; }

    virtual void pre_process();

    virtual void post_process();

    virtual void clear();

    virtual void accept(Patch *p);

    virtual void accept(MeshBlock const *m, std::shared_ptr<DataBlock> const &d);

   private:
    MeshBlock const *m_mesh_;
    std::shared_ptr<AttributeDesc> m_desc_ = nullptr;
    std::shared_ptr<DataBlock> m_data_;
};

class AttributeCollection : public design_pattern::Observable<void(Patch *)> {
    typedef design_pattern::Observable<void(Patch *)> base_type;

   public:
    AttributeCollection(std::shared_ptr<AttributeDict> const &p = nullptr);

    virtual ~AttributeCollection();

    virtual void connect(Attribute *observer);

    virtual void disconnect(Attribute *observer);

    virtual void accept(Patch *p) { base_type::accept(p); }

    template <typename TF>
    void foreach (TF const &fun) {
        design_pattern::Observable<void(Patch *)>::foreach (
            [&](observer_type &obj) { fun(static_cast<Attribute *>(&obj)); });
    }

   private:
    std::shared_ptr<AttributeDict> m_dict_;
};
//
template <typename...>
class AttributeAdapter;

template <typename U>
class AttributeAdapter<U> : public Attribute, public U {
    SP_OBJECT_HEAD(AttributeAdapter<U>, Attribute);

    typedef algebra::traits::value_type_t<U> value_type;

   public:
    template <typename... Args>
    explicit AttributeAdapter(Args &&... args)
        : base_type(AttributeDesc::create<value_type, algebra::traits::iform<U>::value,
                                          algebra::traits::dof<U>::value>(),std::forward<Args>(args)...)
    {
    }

   private:
    struct create_new {};
    template <typename... Args>
    explicit AttributeAdapter(create_new const &, Args &&... args)
        : AttributeAdapter(), U(std::forward<Args>(args)...) {}

   public:
    AttributeAdapter(AttributeAdapter &&) = delete;

    AttributeAdapter(AttributeAdapter const &) = delete;

    ~AttributeAdapter() {}

    using U::operator=;

    virtual std::ostream &print(std::ostream &os, int indent = 0) const {
        return U::print(os, indent);
    }

    virtual void deploy() {
        Attribute::deploy();
        U::deploy();
    }

    template <typename... Args>
    static this_type create(Args &&... args) {
        std::make_shared(create_new(), std::forward<Args>(args)...);
    }

    virtual std::shared_ptr<DataBlock> create_data_block(MeshBlock const *m,
                                                         void *p = nullptr) const {
        return DataBlockAdapter<U>::create(m, static_cast<value_type *>(p));
    };

    virtual void accept(Patch *p) { Attribute::accept(p); }

    virtual void clear() { Attribute::clear(); }

    virtual void pre_process() { Attribute::pre_process(); };

    virtual void post_process() { Attribute::post_process(); }
};
//
template <typename TV, int IFORM = VERTEX, int DOF = 1>
using ArrayAttribute =
    AttributeAdapter<Array<TV, 3 + (((IFORM == VERTEX || IFORM == VOLUME) && DOF == 1) ? 0 : 1)>>;
template <typename TV, typename TM, int IFORM = VERTEX, int DOF = 1>
using FieldAttribute = AttributeAdapter<Field<TV, TM, IFORM, DOF>>;
}
}  // namespace data_block

#endif  // SIMPLA_ATTRIBUTE_H
