//
// Created by salmon on 17-3-21.
//

#ifndef SIMPLA_CONFIGURABLE_H
#define SIMPLA_CONFIGURABLE_H

#include "DataTable.h"
namespace simpla {
namespace data {
class Configurable {
   public:
    explicit Configurable(std::shared_ptr<data::DataTable> const& t = nullptr)
        : m_db_((t != nullptr) ? (t) : std::make_shared<data::DataTable>()) {}
    explicit Configurable(std::string const& s_name) : Configurable() { m_db_->SetValue("name", s_name); }
    explicit Configurable(Configurable const& other) : m_db_(other.m_db_) {}
    virtual ~Configurable() = default;

    virtual void swap(Configurable& other) { std::swap(m_db_, other.m_db_); }

    std::shared_ptr<data::DataTable> const& db() const { return m_db_; }
    std::shared_ptr<data::DataTable>& db() { return m_db_; }
    std::string name() const { return db()->GetValue<std::string>("name", ""); }

   private:
    std::shared_ptr<data::DataTable> m_db_;
};
}
}
#endif  // SIMPLA_CONFIGURABLE_H
