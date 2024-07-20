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

struct DataEntryMemory : public Entry {
    SP_DATA_ENTITY_HEAD(Entry, DataEntryMemory, mem)

   public:
    std::shared_ptr<Entry> Copy() const override;

    size_type Add(const std::shared_ptr<Entry>& v) override;
    using base_type::Set;
    using base_type::Add;
    using base_type::Get;
    std::shared_ptr<Entry> CreateNode(eNodeType e_type) const override;
    size_type size() const override;
    size_type Set(std::string const& uri, std::shared_ptr<Entry> const& v) override;
    size_type Set(index_type s, std::shared_ptr<Entry> const& v) override;
    size_type Add(std::string const& uri, std::shared_ptr<Entry> const& v) override;
    size_type Add(index_type s, std::shared_ptr<Entry> const& v) override;
    size_type Delete(std::string const& s) override;
    size_type Delete(index_type s) override;
    std::shared_ptr<const Entry> Get(std::string const& uri) const override;
    std::shared_ptr<const Entry> Get(index_type s) const override;
    std::shared_ptr<Entry> Get(std::string const& uri) override;
    std::shared_ptr<Entry> Get(index_type s) override;
    void Foreach(std::function<void(std::string const&, std::shared_ptr<Entry> const&)> const& f) override;
    void Foreach(
        std::function<void(std::string const&, std::shared_ptr<const Entry> const&)> const& f) const override;

   private:
    std::map<std::string, std::shared_ptr<Entry>> m_table_;
};
}  // namespace data
}  // namespace simpla
#endif  // SIMPLA_DATANODEMEMORY_H
