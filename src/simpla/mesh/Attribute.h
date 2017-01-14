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

struct AttributeDesc : public concept::Configurable, public Object {
    AttributeDesc() {}

    virtual ~AttributeDesc() {}

    virtual std::type_index value_type_index() const = 0;

    virtual std::type_info const &value_type_info() const = 0;

    virtual int entity_type() const = 0;

    virtual int dof() const = 0;
};

template <typename TV, int IFORM, int DOF>
struct AttributeDescTemp : public AttributeDesc {
    AttributeDescTemp() : AttributeDesc() {}

    virtual ~AttributeDescTemp() {}

    virtual std::type_index value_type_index() const { return std::type_index(value_type_info()); };

    virtual std::type_info const &value_type_info() const { return (typeid(TV)); };

    virtual int entity_type() const { return IFORM; };

    virtual int dof() const { return DOF; };
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

    Attribute(std::shared_ptr<DataBlock> const &d = nullptr,
              std::shared_ptr<AttributeDesc> const &desc = nullptr);

    Attribute(AttributeCollection *p, std::shared_ptr<AttributeDesc> const &desc);

    Attribute(Attribute const &other) = delete;

    Attribute(Attribute &&other) = delete;

    virtual ~Attribute();

    virtual std::ostream &print(std::ostream &os, int indent = 0) const { return os; };

    virtual std::shared_ptr<DataBlock> create_data_block(MeshBlock const *m,
                                                         void *p = nullptr) const = 0;

    virtual std::shared_ptr<AttributeDesc> description() const { return m_desc_; }

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
    explicit AttributeAdapter(Args &&... args) : Attribute(std::forward<Args>(args)...) {}

    //    template <typename... Args>
    //    explicit AttributeAdapter(Args &&... args)
    //        : Attribute(nullptr,
    //                    std::make_shared<AttributeDescTemp<value_type,
    //                    algebra::traits::iform<U>::value,
    //                                                       algebra::traits::dof<U>::value>>(
    //                        std::forward<Args>(args)...)) {}

   private:
    struct create_it {};
    template <typename... Args>
    explicit AttributeAdapter(create_it const &, Args &&... args)
        : U(std::forward<Args>(args)...) {}

   public:
    AttributeAdapter(AttributeAdapter &&) = delete;

    AttributeAdapter(AttributeAdapter const &) = delete;

    ~AttributeAdapter() {}

    using U::operator=;

    virtual std::shared_ptr<AttributeDesc> description() const {
        return std::make_shared<AttributeDescTemp<value_type, algebra::traits::iform<U>::value,
                                                  algebra::traits::dof<U>::value>>();
    }
    template <typename... Args>
    static std::shared_ptr<this_type> create(Args &&... args) {
        std::make_shared(create_it(), std::forward<Args>(args)...);
    }

    virtual std::ostream &print(std::ostream &os, int indent = 0) const {
        return U::print(os, indent);
    }

    virtual std::shared_ptr<DataBlock> create_data_block(MeshBlock const *m,
                                                         void *p = nullptr) const {
        return DataBlockAdapter<U>::create(m, static_cast<value_type *>(p));
    };

    virtual void accept(Patch *p) {
        Attribute::accept(p);
        //        accept(this, Attribute::data());
    }

    virtual void clear() {
        Attribute::clear();
        U::clear();
    }

    virtual void pre_process() {
        Attribute::pre_process();
        //        U::pre_process();
    };

    virtual void post_process() {
        //        U::post_process();
        Attribute::post_process();
    }
    virtual void deploy() {
        U::deploy();
        Attribute::deploy();
    }
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
