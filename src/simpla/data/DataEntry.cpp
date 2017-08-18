//
// Created by salmon on 17-8-18.
//

#include "DataEntry.h"
namespace simpla {
namespace data {
struct DataEntry::pimpl_s {
    std::shared_ptr<DataEntry> m_root_ = nullptr;
    std::shared_ptr<DataEntry> m_parent_ = nullptr;
};
DataEntry::DataEntry() : m_pimpl_(new pimpl_s) {}
DataEntry::~DataEntry() { delete m_pimpl_; }

std::shared_ptr<DataEntry> DataEntry::Root() { return nullptr; }
std::shared_ptr<DataEntry> DataEntry::Parent() { return nullptr; }
std::shared_ptr<DataEntry> DataEntry::Next() const { return nullptr; }
std::shared_ptr<DataEntry> DataEntry::Child(index_type s) const { return nullptr; }
std::shared_ptr<DataEntry> DataEntry::Child(std::string const& uri) const { return nullptr; }

int DataEntry::Delete() { return 0; }

std::shared_ptr<DataEntity> DataEntry::Get() { return nullptr; }
std::shared_ptr<DataEntity> DataEntry::Get() const { return nullptr; }
int DataEntry::Set(const std::shared_ptr<DataEntity>& v) { return 0; }
int DataEntry::Add(const std::shared_ptr<DataEntity>& v) { return 0; }
}
}