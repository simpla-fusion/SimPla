//
// Created by salmon on 16-10-28.
//

#ifndef SIMPLA_LUADATABASE_H
#define SIMPLA_LUADATABASE_H

#include <memory>
#include <ostream>
#include <string>
#include "DataBackend.h"
#include "../../../cmake-build-release/include/simpla/SIMPLA_config.h"

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
    virtual DataBackend* Copy() const;
    virtual bool empty() const;
    virtual void Open(std::string const& url, std::string const& status = "");
    virtual void Parse(std::string const& str);
    virtual void Flush();
    virtual void Close();
    virtual void Clear();
    virtual void Reset();

    DataEntity Get(std::string const& uri);
    bool Put(std::string const& uri, DataEntity&& v);
    bool Post(std::string const& uri, DataEntity&& v);
    size_type Delete(std::string const &uri);
    size_t Count(std::string const& uri) const;

   private:
    struct pimpl_s;
    pimpl_s* m_pimpl_ = nullptr;
};
}  // { namespace data {
}  // namespace simpla
#endif  // SIMPLA_LUADATABASE_H
