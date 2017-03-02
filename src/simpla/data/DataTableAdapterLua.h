//
// Created by salmon on 16-10-28.
//

#ifndef SIMPLA_LUADATABASE_H
#define SIMPLA_LUADATABASE_H

#include <simpla/toolbox/LuaObject.h>
#include "DataTable.h"

namespace simpla {
namespace data {
//class DataEntityAdapter<toolbox::LuaObject> : public DataEntity {
//    toolbox::LuaObject m_lua_obj_;
//
//   public:
//    DataEntityAdapter();
//    DataEntityAdapter(toolbox::LuaObject);
//    virtual ~DataEntityAdapter();
//
//    virtual bool isLight() const { return true; };
//
//    virtual void Copy(DataTable const&) const;
//    virtual std::shared_ptr<DataEntity> Copy() const;
//    virtual std::shared_ptr<DataEntity> Move();
//    virtual void DeepCopy(DataEntity const& other);
//};
//
//class DataTableAdapter<toolbox::LuaObject> : public DataTable {
//    toolbox::LuaObject m_lua_obj_;
//
//   public:
//    DataTableAdapter();
//    DataTableAdapter(toolbox::LuaObject);
//    virtual ~DataTableAdapter() {}
//    virtual std::ostream& Print(std::ostream& os, int indent = 0) const;
//    virtual bool isTable() const;
//    virtual bool empty() const;
//    virtual bool has(std::string const& key) const;
//    virtual void foreach (std::function<void(std::string const& key, DataEntity const&)> const&) const;
//    virtual void foreach (std::function<void(std::string const& key, DataEntity&)> const& fun);
//    virtual std::shared_ptr<DataEntity> find(std::string const& url) const;
//    virtual void Set(std::string const& key, std::shared_ptr<DataEntity> const& v);
//    virtual std::shared_ptr<DataEntity> Get(std::string const& url);
//    virtual std::shared_ptr<DataEntity> Get(std::string const& url) const;
//    virtual DataTable& GetTable(std::string const& url) { return Get(url + ".")->asTable(); }
//    virtual const DataTable& GetTable(std::string const& url) const { return Get(url + ".")->asTable(); }
//};
}
}  // namespace simpla { namespace toolbox {
#endif  // SIMPLA_LUADATABASE_H
