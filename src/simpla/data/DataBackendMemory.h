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
   public:
    DataBackendMemory();
    virtual ~DataBackendMemory();
    virtual std::type_info const& type() const { return typeid(DataBackendMemory); };
    virtual std::ostream& Print(std::ostream& os, int indent = 0) const;
    virtual void Parse(std::string const& str);
    virtual void Open(std::string const& url, std::string const& status = "");
    virtual void Close();
    virtual void Flush();
    virtual bool empty() const;
    virtual void clear();
    virtual void reset();
    virtual DataTable* CreateTable(std::string const& url);
    virtual DataEntity* Set(std::string const& k, std::shared_ptr<DataEntity> const& v);
    virtual DataEntity* Get(std::string const& url);
    virtual DataEntity const* Get(std::string const& url) const;
    virtual bool Erase(std::string const& url);

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};  // class DataBackend {
}  // namespace data {
}  // namespace simpla{
#endif  // SIMPLA_DATABACKENDMEMORY_H
