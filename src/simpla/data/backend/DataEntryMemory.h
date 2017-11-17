//
// Created by salmon on 17-9-1.
//

#ifndef SIMPLA_DATANODEMEMORY_H
#define SIMPLA_DATANODEMEMORY_H

#include <simpla/utilities/Factory.h>
#include "../DataBlock.h"
#include "../DataEntry.h"
#include "../Serializable.h"
namespace simpla {
namespace data {

struct DataEntryMemory : public DataEntry {
    SP_DATA_ENTITY_HEAD(DataEntry, DataEntryMemory, mem)

   public:
    size_type Add(const std::shared_ptr<DataEntry>& v) override;
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

   private:
    std::map<std::string, std::shared_ptr<DataEntry>> m_table_;
};
}  // namespace data
}  // namespace simpla
#endif  // SIMPLA_DATANODEMEMORY_H
