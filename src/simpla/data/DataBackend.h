//
// Created by salmon on 17-3-6.
//

#ifndef SIMPLA_DATABACKEND_H
#define SIMPLA_DATABACKEND_H

#include <simpla/engine/SPObjectHead.h>
#include <simpla/toolbox/Log.h>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include "../../../cmake-build-release/include/simpla/SIMPLA_config.h"

namespace simpla {
namespace data {
class DataEntity;

class DataBackend {
    SP_OBJECT_BASE(DataBackend);

   public:
    DataBackend() {}
    virtual ~DataBackend(){};
    virtual void Flush(){};
    virtual std::ostream& Print(std::ostream& os, int indent = 0) const { return os; };
    virtual std::unique_ptr<DataBackend> CreateNew() const = 0;
    virtual size_type size() const = 0;
    virtual std::shared_ptr<DataEntity> Get(std::string const& URI) const = 0;
    virtual std::shared_ptr<DataEntity> Get(id_type key) const = 0;
    virtual bool Set(std::string const& URI, std::shared_ptr<DataEntity> const&) = 0;
    virtual bool Set(id_type key, std::shared_ptr<DataEntity> const&) = 0;
    virtual bool Add(std::string const& URI, std::shared_ptr<DataEntity> const&) = 0;
    virtual bool Add(id_type key, std::shared_ptr<DataEntity> const&) = 0;
    virtual size_type Delete(std::string const& URI) = 0;
    virtual size_type Delete(id_type key) = 0;
    virtual void DeleteAll() = 0;
    virtual size_type Accept(std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const&) const = 0;
    virtual size_type Accept(std::function<void(id_type, std::shared_ptr<DataEntity>)> const&) const = 0;

   private:
    id_type hash(std::string const&) const;

};  // class DataBackend {
}  // namespace data {
}  // namespace simpla{
#endif  // SIMPLA_DATABACKEND_H
