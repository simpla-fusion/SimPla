//
// Created by salmon on 17-3-6.
//

#ifndef SIMPLA_DATABACKENDMEMORY_H
#define SIMPLA_DATABACKENDMEMORY_H

#include <ostream>
#include <typeindex>
#include "DataBackend.h"
#include "DataEntity.h"
#include "DataTable.h"
namespace simpla {
namespace data {
class DataBackendMemory : public DataBackend {
    SP_OBJECT_HEAD(DataBackendMemory, DataBackend);

   public:
    DataBackendMemory();

    explicit DataBackendMemory(std::string const& uri, std::string const& status = "");
    ~DataBackendMemory() override;

    DataBackendMemory(this_type const& other);
    DataBackendMemory(this_type&& other) noexcept;

    std::shared_ptr<DataBackend> Duplicate() const override;
    std::shared_ptr<DataBackend> CreateNew() const override;
    void Flush() override;
    std::ostream& Print(std::ostream& os, int indent = 0) const override;
    bool isNull() const;  //!< is not initialized

    std::shared_ptr<DataEntity> Get(std::string const& URI) const override;
    void Set(std::string const& URI, std::shared_ptr<DataEntity> const&, bool overwrite = true) override;
    void Add(std::string const& URI, std::shared_ptr<DataEntity> const&) override;
    size_type Delete(std::string const &URI) override;
    size_type size() const override;
    size_type Foreach(std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const& _) const override;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};  // class DataBackend {
}  // namespace data {
}  // namespace simpla{
#endif  // SIMPLA_DATABACKENDMEMORY_H
