//
// Created by salmon on 17-3-6.
//

#ifndef SIMPLA_DATABACKEND_H
#define SIMPLA_DATABACKEND_H

#include <simpla/SIMPLA_config.h>
#include <simpla/concept/Printable.h>
#include <simpla/design_pattern/Factory.h>
#include <simpla/design_pattern/SingletonHolder.h>
#include <simpla/engine/SPObjectHead.h>
#include <simpla/toolbox/Log.h>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <vector>
namespace simpla {
namespace data {
class DataEntity;
class DataTable;
class DataBackend : public concept::Printable, public std::enable_shared_from_this<DataBackend> {
    SP_OBJECT_BASE(DataBackend);

   public:
    DataBackend(){};
    virtual ~DataBackend(){};
    static std::shared_ptr<DataBackend> Create(std::string const& uri);
    virtual void Connect(std::string const& path) { CHECK(path); };
    virtual void Disconnect(){};
    virtual std::shared_ptr<DataBackend> Create() const = 0;
    virtual std::shared_ptr<DataBackend> Clone() const = 0;
    virtual std::string scheme() const = 0;
    virtual void Flush() = 0;
    virtual std::ostream& Print(std::ostream& os, int indent = 0) const { return os; };
    virtual size_type size() const = 0;

    virtual std::shared_ptr<DataEntity> Get(std::string const& URI) const = 0;
    virtual void Set(std::string const& URI, std::shared_ptr<DataEntity> const&) = 0;
    virtual void Add(std::string const& URI, std::shared_ptr<DataEntity> const&) = 0;
    virtual size_type Delete(std::string const& URI) = 0;

    virtual size_type Accept(std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const&) const = 0;

};  // class DataBackend {

class DataBackendFactory : public design_pattern::Factory<std::string, DataBackend>, public concept::Printable {
    typedef design_pattern::Factory<std::string, DataBackend> base_type;

   public:
    DataBackendFactory();
    virtual ~DataBackendFactory();
    std::vector<std::string> RegisteredBackend() const;
    DataBackend* Create(std::string const& scheme);
    void RegisterDefault(){};
};

#define GLOBAL_DATA_BACKEND_FACTORY SingletonHolder<DataBackendFactory>::instance()
}  // namespace data {
}  // namespace simpla{
#endif  // SIMPLA_DATABACKEND_H
