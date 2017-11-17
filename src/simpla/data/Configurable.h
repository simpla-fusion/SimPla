//
// Created by salmon on 17-11-17.
//

#ifndef SIMPLA_CONFIGURABLE_H
#define SIMPLA_CONFIGURABLE_H

#include <simpla/SIMPLA_config.h>
#include <memory>
namespace simpla {
namespace data {
struct DataEntry;
struct Configurable {
   public:
    Configurable();
    Configurable(Configurable const &other);
    virtual ~Configurable();
    id_type GetUUID() const;

    std::shared_ptr<const DataEntry> db() const;
    std::shared_ptr<DataEntry> db();
    void db(std::shared_ptr<DataEntry> const &);

    void SetName(std::string const &);
    std::string GetName() const;

   private:
    std::shared_ptr<DataEntry> m_db_;
    const id_type m_id_;
};
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_CONFIGURABLE_H
