//
// Created by salmon on 17-8-18.
//

#ifndef SIMPLA_DATAENTRY_H
#define SIMPLA_DATAENTRY_H
#import "simpla/SIMPLA_config.h"

#include <memory>
#include <string>

#include "simpla/utilities/ObjectHead.h"

namespace simpla {
namespace data {
class DataEntity;
class DataBase;
struct DataEntry {
   protected:
    DataEntry();
    struct pimpl_s;
    pimpl_s* m_pimpl_ = nullptr;

   public:
    friend DataBase;

    ~DataEntry();
    SP_DEFAULT_CONSTRUCT(DataEntry)

    virtual std::shared_ptr<DataEntry> Root();
    virtual std::shared_ptr<DataEntry> Parent();
    virtual std::shared_ptr<DataEntry> Next() const;

    virtual std::shared_ptr<DataEntry> Child(std::string const& uri) const;
    virtual std::shared_ptr<DataEntry> Child(index_type s) const;

    virtual int Delete();

    virtual std::shared_ptr<DataEntity> Get();
    virtual std::shared_ptr<DataEntity> Get() const;
    virtual int Set(const std::shared_ptr<DataEntity>& v);
    virtual int Add(const std::shared_ptr<DataEntity>& v);
};
}
}
#endif  // SIMPLA_DATAENTRY_H
