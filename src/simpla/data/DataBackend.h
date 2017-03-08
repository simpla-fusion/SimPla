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

    virtual std::unique_ptr<DataBackend> Copy() const { return std::make_unique<DataBackend>(); };
    virtual bool empty() const { return true; };
    virtual void Open(std::string const& url, std::string const& status = "") {}
    virtual void Parse(std::string const& str){};
    virtual void Flush(){};
    virtual void Close(){};
    virtual void Clear(){};
    virtual void Reset(){};

    virtual size_type count() const { return 0; };

    /** as Array */
    virtual std::shared_ptr<DataEntity> Get(size_type idx) const { return std::make_shared<DataEntity>(); }
    virtual bool Set(size_type idx, std::shared_ptr<DataEntity> const&) { return false; }
    virtual bool Add(std::shared_ptr<DataEntity> const&) { return false; }
    virtual int Delete(size_type idx) { return 0; }

    /** as Table */
    virtual std::shared_ptr<DataEntity> Get(std::string const& key) const { return std::make_shared<DataEntity>(); }
    virtual bool Set(std::string const& key, std::shared_ptr<DataEntity> const&) { return false; }
    virtual bool Add(std::string const& key, std::shared_ptr<DataEntity> const&) { return false; }
    virtual size_type Delete(std::string const& key) { return 0; }

    /**
     * @brief
     * @return number of affected entities
     */
    virtual int Accept(std::function<void(std::string const&, std::shared_ptr<DataEntity> const&)> const&) const {
        UNIMPLEMENTED;
        return 0;
    };
    virtual int Accept(std::function<void(std::string const &, std::shared_ptr<DataEntity> &)> const &) {
        UNIMPLEMENTED;
    };

};  // class DataBackend {
}  // namespace data {
}  // namespace simpla{
#endif  // SIMPLA_DATABACKEND_H
