//
// Created by salmon on 17-2-4.
//

#ifndef SIMPLA_ATTRIBUTE_H
#define SIMPLA_ATTRIBUTE_H
#include <simpla/SIMPLA_config.h>

#include <simpla/concept/Configurable.h>
#include <simpla/concept/Object.h>
#include <simpla/concept/Printable.h>
#include <simpla/concept/Serializable.h>
#include <simpla/data/all.h>

namespace simpla {
namespace mesh {

struct Attribute : public Object, public concept::Configurable, public concept::Printable {
    Attribute() : m_value_type_index_(std::type_index(typeid(Real))), m_iform_(VERTEX), m_dof_(1) { deploy(); }

    Attribute(Attribute const &other) = delete;

    Attribute(Attribute &&other) = delete;

    virtual ~Attribute() {}

    virtual void deploy() {
        m_value_type_index_ = value_type_index();
        m_iform_ = entity_type();
        m_dof_ = dof();
    }

    template <typename ValueType, int IFORM, int DOF>
    static std::shared_ptr<Attribute> create() {
        auto res = std::make_shared<Attribute>();
        res->m_iform_ = IFORM;
        res->m_dof_ = DOF;
        res->m_value_type_index_ = std::type_index(typeid(ValueType));
        return res;
    };
    template <typename ValueType, int IFORM, int DOF>
    static std::shared_ptr<Attribute> create(std::initializer_list<data::KeyValue> const &param) {
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
};  // class Attribute

class AttributeDict : public concept::Printable {
   public:
    virtual std::ostream &Print(std::ostream &os, int indent = 0) const;
    std::pair<std::shared_ptr<Attribute>, bool> register_attr(std::shared_ptr<Attribute> const &desc);
    void erase(id_type const &id);
    void erase(std::string const &id);
    std::shared_ptr<Attribute> find(id_type const &id);
    std::shared_ptr<Attribute> find(std::string const &id);
    std::shared_ptr<Attribute> const &get(std::string const &k) const;
    std::shared_ptr<Attribute> const &get(id_type k) const;

   private:
    std::map<std::string, id_type> m_key_id_;
    std::map<id_type, std::shared_ptr<Attribute>> m_map_;
};  // AttributeDict
}
}
#endif  // SIMPLA_ATTRIBUTE_H
