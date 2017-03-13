//
// Created by salmon on 16-10-28.
//

#ifndef SIMPLA_LUADATABASE_H
#define SIMPLA_LUADATABASE_H
#include <simpla/SIMPLA_config.h>
#include <memory>
#include <ostream>
#include <string>
#include "../DataBackend.h"

namespace simpla {
namespace data {

class DataEntity;

class DataBackendLua : public DataBackend {
    SP_OBJECT_HEAD(DataBackendLua, DataBackend)
   public:
    static constexpr char scheme_tag[] = "lua";

    DataBackendLua();
    DataBackendLua(DataBackendLua const&);
    virtual ~DataBackendLua();
    virtual std::ostream& Print(std::ostream& os, int indent = 0) const;
    virtual std::string scheme() const;
    virtual void Connect(std::string const&);
    virtual void Disconnect();
    virtual std::shared_ptr<DataBackend> Duplicate() const;
    virtual std::shared_ptr<DataBackend> CreateNew() const;
    virtual bool isNull() const;
    virtual void Flush();

    virtual std::shared_ptr<DataEntity> Get(std::string const& URI) const;
    virtual std::shared_ptr<DataEntity> Get(id_type key) const;
    virtual void Set(std::string const& URI, std::shared_ptr<DataEntity> const&);
    virtual void Add(std::string const& URI, std::shared_ptr<DataEntity> const&);
    virtual size_type Delete(std::string const& URI);
    virtual size_type size() const;
    virtual size_type Accept(std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const&) const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

}  // { namespace data {
}  // namespace simpla
#endif  // SIMPLA_LUADATABASE_H
