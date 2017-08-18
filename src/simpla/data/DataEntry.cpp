//
// Created by salmon on 17-8-18.
//

#include "DataEntry.h"
namespace simpla {
namespace data {

bool DataEntry::isNull() const {}
size_type DataEntry::Count() const {}

std::shared_ptr<DataEntity> DataEntry::Get(std::string const& key) {}
std::shared_ptr<DataEntity> DataEntry::Get(std::string const& key) const {}
int DataEntry::Set(std::string const& uri, const std::shared_ptr<DataEntity>& v) {}
int DataEntry::Add(std::string const& uri, const std::shared_ptr<DataEntity>& v) {}
int DataEntry::Delete(std::string const& uri) {}
}
}