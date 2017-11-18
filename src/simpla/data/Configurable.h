//
// Created by salmon on 17-11-17.
//

#ifndef SIMPLA_CONFIGURABLE_H
#define SIMPLA_CONFIGURABLE_H

#include <simpla/SIMPLA_config.h>
#include <memory>
#include "DataEntry.h"
namespace simpla {
namespace data {

#define SP_PROPERTY(_TYPE_, _NAME_)                                                      \
   public:                                                                               \
    void Set##_NAME_(_TYPE_ const &_v_) { this->db()->SetValue(__STRING(_NAME_), _v_); } \
    _TYPE_ Get##_NAME_() const { return this->db()->GetValue<_TYPE_>(__STRING(_NAME_)); }

struct Configurable {
   public:
    Configurable();
    Configurable(Configurable const &other);
    Configurable(Configurable &&other) noexcept;
    virtual ~Configurable();

    std::shared_ptr<const DataEntry> db() const;
    virtual std::shared_ptr<DataEntry> db();
    void SetDB(std::shared_ptr<DataEntry> const &d);

    SP_PROPERTY(std::string, Name);
    SP_PROPERTY(id_type, UUID);

   private:
    std::shared_ptr<DataEntry> m_db_;
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_CONFIGURABLE_H
