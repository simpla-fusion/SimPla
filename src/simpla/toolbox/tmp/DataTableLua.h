//
// Created by salmon on 16-10-28.
//

#ifndef SIMPLA_LUADATABASE_H
#define SIMPLA_LUADATABASE_H

#include <simpla/data/DataEntity.h>
#include <simpla/data/DataTable.h>
#include "simpla/toolbox/LuaObject.h"
namespace simpla {
namespace data {
class DataEntityLua : public DataEntity {
    toolbox::LuaObject m_lua_obj_;
    SP_OBJECT_HEAD(DataEntityLua, DataEntity);

   public:
    DataEntityLua(toolbox::LuaObject const& u);
    DataEntityLua(toolbox::LuaObject&& u);
    ~DataEntityLua();

    std::ostream& Print(std::ostream& os, int indent = 0) const;
    bool isLight() const;

    template <typename U>
    U GetValue() const {
        return m_lua_obj_.as<U>();
    }
    std::shared_ptr<DataEntity> Copy() const;
};
class DataTableLua : public DataTable {
    toolbox::LuaObject m_lua_obj_;

   public:
    DataTableLua();
    DataTableLua(DataTableLua const&);
    DataTableLua(toolbox::LuaObject const&);
    DataTableLua(toolbox::LuaObject&&);
    virtual ~DataTableLua();
    void ParseFile(std::string const& filename);
    void Parse(std::string const& str);
    virtual std::ostream& Print(std::ostream& os, int indent = 0) const;
    virtual bool isTable() const;
    virtual bool empty() const;
    virtual bool has(std::string const& key) const;

    //    virtual std::shared_ptr<DataEntity> find(std::string const& url) const;
    //    virtual void Merge(DataTable const&);
    virtual void Set(std::string const& key, std::shared_ptr<DataEntity> const& v);
    virtual std::shared_ptr<DataEntityLua> GetLua(std::string const& url);
    virtual std::shared_ptr<DataEntityLua> GetLua(std::string const& url) const;
    virtual std::shared_ptr<DataEntity> Get(std::string const& url);
    virtual std::shared_ptr<DataEntity> Get(std::string const& url) const;
    virtual std::shared_ptr<DataTable> CreateTable(std::string const& url);

    template <typename U>
    U GetValue(std::string const& url) const {
        return GetLua(url)->GetValue<U>();
    }

    //    virtual void foreach (std::function<void(std::string const& key, DataEntity const&)> const&) const;
    //    virtual void foreach (std::function<void(std::string const& key, DataEntity&)> const& fun);
};
}
}  // namespace simpla { namespace toolbox {
#endif  // SIMPLA_LUADATABASE_H
