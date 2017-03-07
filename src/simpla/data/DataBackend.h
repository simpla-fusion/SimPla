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

namespace simpla {
namespace data {
class DataEntity;

class DataBackend {
    SP_OBJECT_BASE(DataBackend);

   public:
    DataBackend() {}
    virtual ~DataBackend(){};
    virtual std::ostream& Print(std::ostream& os, int indent = 0) const = 0;
    virtual DataBackend* Copy() const = 0;
    virtual bool empty() const = 0;
    virtual void Open(std::string const& url, std::string const& status = "") = 0;
    virtual void Parse(std::string const& str) = 0;
    virtual void Flush() = 0;
    virtual void Close() = 0;
    virtual void Clear() = 0;
    virtual void Reset() = 0;
    virtual bool Erase(std::string const& k) = 0;
    virtual std::pair<DataEntity*, bool> Insert(std::string const& k) = 0;
    virtual std::pair<DataEntity*, bool> Insert(std::string const& k, DataEntity const& v,
                                                bool assign_is_exists = true) = 0;
    virtual DataEntity* Find(std::string const& url) const = 0;



};  // class DataBackend {
}  // namespace data {
}  // namespace simpla{
#endif  // SIMPLA_DATABACKEND_H
