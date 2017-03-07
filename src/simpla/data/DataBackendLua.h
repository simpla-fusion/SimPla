//
// Created by salmon on 16-10-28.
//

#ifndef SIMPLA_LUADATABASE_H
#define SIMPLA_LUADATABASE_H

#include <memory>
#include <ostream>
#include <string>
#include "DataBackend.h"
namespace simpla {
namespace data {

class DataTable;
class DataEntity;

class DataBackendLua : public DataBackend {
   public:
    DataBackendLua(DataBackendLua const&);
    DataBackendLua(std::string const& url = "", std::string const& status = "");
    virtual ~DataBackendLua();
    virtual std::type_info const& type() const { return typeid(DataBackendLua); };
    virtual std::ostream& Print(std::ostream& os, int indent = 0) const;

    virtual void Open(std::string const& url, std::string const& status = "");
    virtual void Parse(std::string const& str);
    virtual void Flush();
    virtual void Close();
    virtual bool empty() const;
    virtual void Clear();
    virtual void Reset();
    virtual DataBackend* Copy() const;

    virtual bool Erase(std::string const& k);
    virtual std::pair<DataEntity*, bool> Insert(std::string const& k);
    virtual std::pair<DataEntity*, bool> Insert(std::string const& k, DataEntity const& v,
                                                bool assign_is_exists = true);
    virtual DataEntity* Find(std::string const& url) const;

   private:
    struct pimpl_s;
    pimpl_s* m_pimpl_ = nullptr;
};
}  // { namespace data {
}  // namespace simpla
#endif  // SIMPLA_LUADATABASE_H
