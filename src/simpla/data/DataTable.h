//
// Created by salmon on 16-10-6.
//

#ifndef SIMPLA_DATATABLE_H_
#define SIMPLA_DATATABLE_H_

#include <simpla/SIMPLA_config.h>
#include <simpla/engine/SPObjectHead.h>
#include <memory>
#include "DataEntity.h"
#include "DataTraits.h"
namespace simpla {
namespace data {
template <typename U, typename Enable = void>
class DataTableWrapper {};
class DataBackend;
/** @ingroup data */
/**
 * @brief  a @ref DataEntity tree, a key-value table of @ref DataEntity, which is similar as Group
 * in HDF5, but all node/table are DataEntity.
 * @design_pattern
 *  - Proxy of DataBeckend
 */
class DataTable : public DataEntity {
    SP_OBJECT_HEAD(DataTable, DataEntity);
    std::shared_ptr<DataBackend> m_backend_ = nullptr;

   public:
    DataTable(std::shared_ptr<DataBackend> const& p = nullptr);
    DataTable(std::string const& url, std::string const& status = "");
    DataTable(const DataTable&);
    DataTable(DataTable&&);
    virtual ~DataTable();

    virtual bool isTable() const { return true; }
    std::type_info const& type() const { return typeid(DataTable); };
    std::shared_ptr<DataTable> Copy() const;

    std::shared_ptr<DataBackend> backend() const { return m_backend_; }

    virtual std::ostream& Print(std::ostream& os, int indent = 0) const;
    virtual void Parse(std::string const& str);
    virtual void Open(std::string const& url, std::string const& status = "");
    virtual void Flush();
    virtual void Close();
    virtual void Clear();
    virtual void Reset();

    virtual bool empty() const;
    virtual size_type count() const;

    virtual std::shared_ptr<DataEntity> Get(std::string const& key) const;
    virtual bool Set(DataTable const& other);
    virtual bool Set(std::string const& key, std::shared_ptr<DataEntity> const&);
    virtual bool Add(std::string const& key, std::shared_ptr<DataEntity> const&);
    virtual size_type Delete(std::string const& key);

    template <typename U>
    bool Set(std::string const& uri, U const& v) {
        return Set(uri, make_data_entity(v));
    };
    template <typename U>
    bool Set(std::string const& uri, std::initializer_list<U> const& u) {
        return Set(uri, make_data_entity(u));
    };
    template <typename U>
    bool Set(std::string const& uri, std::initializer_list<std::initializer_list<U>> const& u) {
        return Set(uri, make_data_entity(u));
    };
    template <typename U>
    bool Set(std::string const& uri, std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& u) {
        return Set(uri, make_data_entity(u));
    };
    template <typename U>
    bool Add(std::string const& uri, U const& v) {
        return Add(uri, DataEntity{v});
    };

    template <typename U>
    bool Add(std::string const& uri, U&& v) {
        return Set(uri, DataEntity{v});
    };

    template <typename U>
    bool Add(U const& u) {
        return Add(make_data_entity(u));
    };
};

}  // namespace data
}  // namespace simpla

#endif  // SIMPLA_DATATABLE_H_
