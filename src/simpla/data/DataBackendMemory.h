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
    virtual std::ostream& Print(std::ostream& os, int indent = 0) const;
    virtual DataBackend* Copy() const;
    virtual bool empty() const;
    virtual void Open(std::string const& url, std::string const& status = "");
    virtual void Parse(std::string const& str);
    virtual void Flush();
    virtual void Close();
    virtual void Clear();
    virtual void Reset();

    virtual DataEntity Get(std::string const& uri);
    virtual bool Put(std::string const& uri, DataEntity&& v);
    virtual bool Post(std::string const& uri, DataEntity&& v);
    virtual size_type Delete(std::string const& uri);
    virtual size_type Count(std::string const& uri) const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};  // class DataBackend {
}  // namespace data {
}  // namespace simpla{
#endif  // SIMPLA_DATABACKENDMEMORY_H
