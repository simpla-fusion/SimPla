//
// Created by salmon on 17-3-21.
//

#ifndef SIMPLA_CONFIGURABLE_H
#define SIMPLA_CONFIGURABLE_H

#include <memory>
#include <string>
#include "DataTable.h"
namespace simpla {
namespace data {
class Configurable {
   public:
    explicit Configurable(std::shared_ptr<data::DataTable> const& t = nullptr)
        : m_db_((t != nullptr) ? (t) : std::make_shared<data::DataTable>()) {}

    Configurable(Configurable const& other) = default;
    Configurable(Configurable&& other) noexcept = default;
    Configurable& operator=(Configurable const& other) {
        Configurable(other).swap(*this);
        return *this;
    };
    Configurable& operator=(Configurable&& other) noexcept {
        Configurable(other).swap(*this);
        return *this;
    }
    virtual ~Configurable() = default;

    virtual void swap(Configurable& other) { std::swap(m_db_, other.m_db_); }

    std::shared_ptr<data::DataTable> const& db() const { return m_db_; }
    std::shared_ptr<data::DataTable>& db() { return m_db_; }
    std::string name() const { return db()->GetValue<std::string>("name", ""); }

   private:
    std::shared_ptr<data::DataTable> m_db_ = nullptr;
};
}
}
#endif  // SIMPLA_CONFIGURABLE_H
