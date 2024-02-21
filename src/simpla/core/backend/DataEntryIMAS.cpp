//
// Created by salmon on 17-8-24.
//
#include <simpla/utilities/Factory.h>
#include "../DataEntry.h"
#include "../Serializable.h"

namespace simpla {
namespace data {

struct DataEntryIMAS : public DataEntry {
    SP_DATA_ENTITY_HEAD(DataEntry, DataEntryIMAS, imas)

    int Connect(std::string const& authority, std::string const& path, std::string const& query,
                std::string const& fragment) override;
    int Disconnect() override;
    bool isValid() const override;

    using base_type::Set;
    using base_type::Add;
    using base_type::Get;
    std::shared_ptr<DataEntry> CreateNode(eNodeType e_type) const override;
    size_type size() const override;
    size_type Set(std::string const& uri, std::shared_ptr<DataEntry> const& v) override;
    size_type Set(index_type s, std::shared_ptr<DataEntry> const& v) override;
    size_type Add(std::string const& uri, std::shared_ptr<DataEntry> const& v) override;
    size_type Add(index_type s, std::shared_ptr<DataEntry> const& v) override;
    size_type Delete(std::string const& s) override;
    size_type Delete(index_type s) override;
    std::shared_ptr<const DataEntry> Get(std::string const& uri) const override;
    std::shared_ptr<const DataEntry> Get(index_type s) const override;
    std::shared_ptr<DataEntry> Get(std::string const& uri) override;
    std::shared_ptr<DataEntry> Get(index_type s) override;
    void Foreach(std::function<void(std::string const&, std::shared_ptr<DataEntry> const&)> const& f) override;
    void Foreach(
        std::function<void(std::string const&, std::shared_ptr<const DataEntry> const&)> const& f) const override;
};
SP_REGISTER_CREATOR(DataEntry, DataEntryIMAS);
DataEntryIMAS::DataEntryIMAS(DataEntry::eNodeType e_type) : base_type(e_type){};
DataEntryIMAS::DataEntryIMAS(DataEntryIMAS const& other) = default;
DataEntryIMAS::~DataEntryIMAS() = default;

int DataEntryIMAS::Connect(std::string const& authority, std::string const& path, std::string const& query,
                           std::string const& fragment) {
    return 0;
}
int DataEntryIMAS::Disconnect() { return 0; }
bool DataEntryIMAS::isValid() const { return false; }

size_type DataEntryIMAS::size() const { return 0; }
std::shared_ptr<DataEntry> DataEntryIMAS::CreateNode(DataEntry::eNodeType e_type) const { return nullptr; }

size_type DataEntryIMAS::Set(std::string const& uri, const std::shared_ptr<DataEntry>& v) { return 0; }
size_type DataEntryIMAS::Set(index_type s, std::shared_ptr<DataEntry> const& v) { return 0; }
size_type DataEntryIMAS::Add(std::string const& uri, const std::shared_ptr<DataEntry>& v) { return 0; }
size_type DataEntryIMAS::Add(index_type s, std::shared_ptr<DataEntry> const& v) { return 0; }
size_type DataEntryIMAS::Delete(std::string const& s) { return 0; }
size_type DataEntryIMAS::Delete(index_type s) { return 0; }
std::shared_ptr<DataEntry> DataEntryIMAS::Get(index_type s) { return nullptr; }
std::shared_ptr<const DataEntry> DataEntryIMAS::Get(index_type s) const { return nullptr; }
std::shared_ptr<DataEntry> DataEntryIMAS::Get(std::string const& uri) { return 0; }
std::shared_ptr<const DataEntry> DataEntryIMAS::Get(std::string const& uri) const { return 0; }
void DataEntryIMAS::Foreach(
    std::function<void(std::string const&, std::shared_ptr<const DataEntry> const&)> const& f) const {
    UNIMPLEMENTED;
}
void DataEntryIMAS::Foreach(std::function<void(std::string const&, std::shared_ptr<DataEntry> const&)> const& f) {
    UNIMPLEMENTED;
}

}  // { namespace data {
}  // namespace simpla