//
// Created by salmon on 17-3-6.
//

#ifndef SIMPLA_DATABACKEND_H
#define SIMPLA_DATABACKEND_H

#include <memory>
#include <typeinfo>
namespace simpla {
namespace data {
class DataEntity;
class DataTable;
class DataBackend {
   public:
    DataBackend() {}
    virtual ~DataBackend(){};
    virtual std::type_info const& type() const = 0;
    virtual std::ostream& Print(std::ostream& os, int indent = 0) const = 0;

    virtual void Parse(std::string const& str) = 0;
    virtual void Open(std::string const& url, std::string const& status = "") = 0;
    virtual void Flush() = 0;
    virtual void Close() = 0;

    virtual bool empty() const = 0;
    virtual void clear() = 0;
    virtual void reset() = 0;

    virtual DataTable * CreateTable(std::string const &url) = 0;
    virtual bool Erase(std::string const& k) = 0;
    virtual DataEntity* Set(std::string const& k, std::shared_ptr<DataEntity> const& v) = 0;
    virtual DataEntity* Get(std::string const& url) = 0;
    virtual DataEntity const* Get(std::string const& url) const = 0;

};  // class DataBackend {
}  // namespace data {
}  // namespace simpla{
#endif  // SIMPLA_DATABACKEND_H
