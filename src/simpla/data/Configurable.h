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
    Configurable() {}
    Configurable(Configurable const& other) : m_db_(other.m_db_){};
    Configurable(Configurable&& other) noexcept : m_db_(other.m_db_){};

    Configurable& operator=(Configurable const& other) {
        Configurable(other).swap(*this);
        return *this;
    };
    Configurable& operator=(Configurable&& other) noexcept {
        Configurable(other).swap(*this);
        return *this;
    }
    virtual ~Configurable() = default;

    virtual void swap(Configurable& other) { m_db_.swap(other.m_db_); }

    const data::DataTable& db() const { return m_db_; }
    data::DataTable& db() { return m_db_; }
    template <typename U>
    U GetProperty(std::string const& uri) const {
        return m_db_.template GetValue<U>(uri);
    }
    template <typename U>
    U GetProperty(std::string const& uri, U const& default_value) const {
        return m_db_.template GetValue<U>(uri, default_value);
    }
    template <typename U>
    void SetProperty(std::string const& uri, U const& value) {
        m_db_.SetValue(uri, value);
    }

   private:
    data::DataTable m_db_{};
};
}
}
#endif  // SIMPLA_CONFIGURABLE_H
