//
// Created by salmon on 17-3-21.
//

#ifndef SIMPLA_CONFIGURABLE_H
#define SIMPLA_CONFIGURABLE_H

#include <simpla/data/DataTable.h>
namespace simpla {
namespace concept {
class Configurable {
   public:
    Configurable(std::shared_ptr<data::DataEntity> const& t = nullptr) {
        m_db_ = (t != nullptr && t->isTable()) ? std::dynamic_pointer_cast<data::DataTable>(t)
                                               : std::make_shared<data::DataTable>();

        if (t != nullptr && t->isLight() && t->value_type_info() == typeid(std::string)) {
            m_db_->SetValue("name", data::data_cast<std::string>(*t));
        }
    }
    virtual ~Configurable() {}
    std::shared_ptr<data::DataTable> db() const { return m_db_; }
    std::shared_ptr<data::DataTable> db() { return m_db_; }
    std::string name() const { return db()->GetValue<std::string>("name", ""); }

   private:
    std::shared_ptr<data::DataTable> m_db_;
};
}
}
#endif  // SIMPLA_CONFIGURABLE_H
