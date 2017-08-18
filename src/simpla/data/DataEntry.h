//
// Created by salmon on 17-8-18.
//

#ifndef SIMPLA_DATAENTRY_H
#define SIMPLA_DATAENTRY_H

#include <memory>
#include <string>
#import "simpla/SIMPLA_config.h"
namespace simpla {
namespace data {
class DataEntity;

struct DataEntry {
   protected:
    DataEntry();

   public:
    std::shared_ptr<DataEntry> New();
    virtual std::shared_ptr<DataEntry> GetRoot();

    virtual bool isNull() const;
    virtual size_type Count() const;

    virtual std::shared_ptr<DataEntity> Get(std::string const& key);
    virtual std::shared_ptr<DataEntity> Get(std::string const& key) const;
    virtual int Set(std::string const& uri, const std::shared_ptr<DataEntity>& v);
    virtual int Add(std::string const& uri, const std::shared_ptr<DataEntity>& v);
    virtual int Delete(std::string const& uri);
};
}
}
#endif  // SIMPLA_DATAENTRY_H
