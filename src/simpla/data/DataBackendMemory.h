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
    DataBackendMemory(std::string const& url = "", std::string const& status = "");
    DataBackendMemory(const DataBackendMemory&);
    virtual ~DataBackendMemory();
    virtual std::unique_ptr<DataBackend> Copy() const;
    virtual bool empty() const;
    virtual void Open(std::string const& url, std::string const& status = "");
    virtual void Parse(std::string const& str);
    virtual void Flush();
    virtual void Close();
    virtual void Clear();
    virtual void Reset();
    virtual size_type count() const;
    virtual std::shared_ptr<DataEntity> Get(std::string const& key) const;
    virtual bool Set(std::string const& key, std::shared_ptr<DataEntity> const&);
    virtual bool Add(std::string const& key, std::shared_ptr<DataEntity> const&);
    virtual size_type Delete(std::string const& key);
    virtual size_type Count(std::string const& uri) const;

    virtual size_type Accept(std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const&) const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};  // class DataBackend {
}  // namespace data {
}  // namespace simpla{
#endif  // SIMPLA_DATABACKENDMEMORY_H
