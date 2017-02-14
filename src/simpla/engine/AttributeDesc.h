//
// Created by salmon on 17-2-4.
//

#ifndef SIMPLA_ATTRIBUTE_H
#define SIMPLA_ATTRIBUTE_H
#include <simpla/SIMPLA_config.h>

#include <simpla/concept/Configurable.h>
#include <simpla/concept/Printable.h>
#include <simpla/concept/Serializable.h>
#include <simpla/data/all.h>
#include <simpla/engine/Object.h>

namespace simpla {
namespace engine {

struct AttributeDesc : public Object {
   public:
    AttributeDesc(std::string const &name_str, std::type_index const &v_idx = std::type_index(typeid(Real)),
                  int IFORM = VERTEX, int DOF = 1)
        : m_name_(name_str), m_value_type_index_(v_idx), m_iform_(IFORM), m_dof_(DOF) {}
    AttributeDesc(AttributeDesc const &other) = delete;
    AttributeDesc(AttributeDesc &&other) = delete;
    ~AttributeDesc() {}
    std::string const &name() const { return m_name_; }
    std::type_index value_type_index() const { return m_value_type_index_; };
    int entity_type() const { return m_iform_; };
    int dof() const { return m_dof_; };
    template <typename U>
    bool check_value_type() const {
        return std::type_index(typeid(U)) == m_value_type_index_;
    }

   private:
    std::string m_name_ = "unnamed";
    std::type_index m_value_type_index_;
    int m_iform_ = VERTEX;
    int m_dof_ = 1;
};  // class AttributeDesc

class AttributeDict : public concept::Printable {
   public:
    virtual std::ostream &Print(std::ostream &os, int indent = 0) const;
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
};  // AttributeDict
}  // namespace engine {
}  // namespace simpla {

#endif  // SIMPLA_ATTRIBUTE_H
