//
// Created by salmon on 16-10-28.
//

#ifndef SIMPLA_LUADATABASE_H
#define SIMPLA_LUADATABASE_H

#include <memory>
#include <ostream>
#include <string>
#include "../../../cmake-build-release/include/simpla/SIMPLA_config.h"
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
    virtual std::unique_ptr<DataBackend> Copy() const;
    virtual bool empty() const;
    virtual void Open(std::string const& url, std::string const& status = "");
    virtual void Parse(std::string const& str);
    virtual void Flush();
    virtual void Close();
    virtual void Clear();
    virtual void Reset();

    virtual std::shared_ptr<DataEntity> Get(std::string const& key) const;
    virtual bool Set(std::string const& key, std::shared_ptr<DataEntity> const&);
    virtual bool Add(std::string const& key, std::shared_ptr<DataEntity> const&);
    virtual size_type Delete(std::string const& key);
    virtual size_type Count(std::string const& uri) const;

   private:
    struct pimpl_s;
    pimpl_s* m_pimpl_ = nullptr;
};
}  // { namespace data {
}  // namespace simpla
#endif  // SIMPLA_LUADATABASE_H
