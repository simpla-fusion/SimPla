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
    virtual std::ostream& Print(std::ostream& os, int indent = 0) const { return os; }

    virtual std::unique_ptr<DataBackend> Copy() const = 0;
    virtual bool empty() const = 0;
    virtual void Open(std::string const& url, std::string const& status = "") = 0;
    virtual void Parse(std::string const& str) = 0;
    virtual void Flush() = 0;
    virtual void Close() = 0;
    virtual void Clear() = 0;
    virtual void Reset() = 0;

    virtual size_type count() const = 0;

    /** as Array */
    //    virtual std::shared_ptr<DataEntity> Get(size_type idx) const = 0;
    //    virtual bool Set(size_type idx, std::shared_ptr<DataEntity> const&) = 0;
    //    virtual bool Add(std::shared_ptr<DataEntity> const&) = 0;
    //    virtual int Delete(size_type idx) = 0;

    /** as Table */
    virtual std::shared_ptr<DataEntity> Get(std::string const& key) const = 0;
    virtual bool Set(std::string const& key, std::shared_ptr<DataEntity> const&) = 0;
    virtual bool Add(std::string const& key, std::shared_ptr<DataEntity> const&) = 0;
    virtual size_type Delete(std::string const& key) = 0;

    /**
     * @brief
     * @return number of affected entities
     */
    virtual size_type Accept(std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const&) const = 0;

};  // class DataBackend {
}  // namespace data {
}  // namespace simpla{
#endif  // SIMPLA_DATABACKEND_H
