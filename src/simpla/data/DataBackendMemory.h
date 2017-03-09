//
// Created by salmon on 17-3-6.
//

#ifndef SIMPLA_DATABACKENDMEMORY_H
#define SIMPLA_DATABACKENDMEMORY_H

#include <ostream>
#include <typeindex>
#include "DataBackend.h"
#include "DataEntity.h"
#include "DataTable.h"
namespace simpla {
namespace data {
class DataBackendMemory : public DataBackend {
    SP_OBJECT_HEAD(DataBackendMemory, DataBackend);

   public:
    DataBackendMemory(std::string const& url = "", std::string const& status = "");
    DataBackendMemory(const DataBackendMemory&);
    DataBackendMemory(DataBackendMemory&&);

    virtual ~DataBackendMemory();

    virtual std::unique_ptr<DataBackend> CreateNew() const;
    //    virtual void Initialize();
    //    virtual void Finalize();
    //    virtual void Flush();
    virtual bool IsNull() const;  //!< is not initialized

    virtual std::shared_ptr<DataEntity> Get(std::string const& URI) const;
    virtual std::shared_ptr<DataEntity> Get(id_type key) const;
    virtual bool Set(std::string const& URI, std::shared_ptr<DataEntity> const&);
    virtual bool Set(id_type key, std::shared_ptr<DataEntity> const&);
    virtual bool Add(std::string const& URI, std::shared_ptr<DataEntity> const&);
    virtual bool Add(id_type key, std::shared_ptr<DataEntity> const&);
    virtual size_type Delete(std::string const& URI);
    virtual size_type Delete(id_type key);
    virtual void DeleteAll();
    virtual size_type Count(std::string const& uri = "") const;
    virtual size_type Accept(std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const&) const;
    virtual size_type Accept(std::function<void(id_type, std::shared_ptr<DataEntity>)> const&) const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};  // class DataBackend {
}  // namespace data {
}  // namespace simpla{
#endif  // SIMPLA_DATABACKENDMEMORY_H
