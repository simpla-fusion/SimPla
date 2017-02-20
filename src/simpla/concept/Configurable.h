//
// Created by salmon on 16-11-17.
//

#ifndef SIMPLA_CONFIGURABLE_H
#define SIMPLA_CONFIGURABLE_H

#include <simpla/data/DataTable.h>

namespace simpla {
namespace concept {
/**  @ingroup concept   */

/**
 * @brief a type whose instances has member DataEntityTable db; *
 * @details
 * ## Summary
 * Requirements for a type whose instances share ownership between multiple objects;
 *
 * ## Requirements
 *  Class \c R implementing the concept of @ref Configurable must define:
 *   Pseudo-Signature                                      | Semantics
 *	 ------------------------------------------------------|----------
 * 	 \code   R()                                  \endcode | constructor;
 * 	 \code  virtual ~R()                          \endcode | Destructor
 * 	 \code data::DataEntityTable db               \endcode |
 *   \code std::string name() const               \endcode | if key-value 'name' return it else
 *return empty string
 *
 */

struct Configurable {
    Configurable() {}
    virtual ~Configurable() {}

    inline std::string const &name() const { return m_name_; }
    inline void name(std::string const &s) { m_name_ = s; }
    data::DataTable &db() {
        Click();
        return m_db_;
    };

    data::DataTable const &db() const { return m_db_; };
    void Click() { ++m_click_count_; }
    bool isUpdated() const { return m_current_click_count_ == m_click_count_; }
    virtual void Update() { m_current_click_count_ = m_click_count_; }

   private:
    data::DataTable m_db_;
    std::string m_name_ = "";
    size_type m_click_count_ = 0;
    size_type m_current_click_count_ = 0;
};
}  // namespace concept
}  // namespace  simpla::

#endif  // SIMPLA_CONFIGURABLE_H
