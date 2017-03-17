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
    static constexpr char scheme_tag[] = "mem";

   public:
    DataBackendMemory();
    DataBackendMemory(std::string const& uri, std::string const& status = "");
    DataBackendMemory(const DataBackendMemory&);
    DataBackendMemory(DataBackendMemory&&);

    virtual ~DataBackendMemory();
    virtual std::shared_ptr<DataBackend> Duplicate() const;
    virtual std::shared_ptr<DataBackend> CreateNew() const;
    virtual void Flush();
    virtual std::ostream& Print(std::ostream& os, int indent = 0) const;
    virtual bool isNull() const;  //!< is not initialized

    virtual std::shared_ptr<DataEntity> Get(std::string const& URI) const;
    virtual std::pair<std::shared_ptr<DataEntity>, bool>  Set(std::string const& URI, std::shared_ptr<DataEntity> const&,
                                            bool overwrite = true);
    virtual std::shared_ptr<DataEntity> Add(std::string const& URI, std::shared_ptr<DataEntity> const&);
    virtual size_type Delete(std::string const& URI);
    virtual size_type size() const;

    virtual size_type Foreach(std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const&) const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};  // class DataBackend {
}  // namespace data {
}  // namespace simpla{
#endif  // SIMPLA_DATABACKENDMEMORY_H
