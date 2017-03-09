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
class KeyValue;
/** @ingroup data */
/**
 * @brief  a @ref DataEntity tree, a key-value table of @ref DataEntity, which is similar as Group
 * in HDF5, but all node/table are DataEntity.
 * @design_pattern
 *  - Proxy of DataBackend
 */
class DataTable : public DataEntity {
    SP_OBJECT_HEAD(DataTable, DataEntity);
    std::unique_ptr<DataBackend> m_backend_;

   public:
    DataTable(std::string const& url = "", std::string const& status = "");
    DataTable(std::unique_ptr<DataBackend>&& p);
    DataTable(const DataTable&);
    DataTable(DataTable&&);
    ~DataTable() final;

    void swap(DataTable&);

    //******************************************************************************************************************
    /** Interface DataEntity */
    std::ostream& Print(std::ostream& os, int indent = 0) const;
    bool isTable() const { return true; }
    std::type_info const& type() const { return typeid(DataTable); };
    std::shared_ptr<DataEntity> Copy() const;
    DataBackend const* backend() const { return m_backend_.get(); }

    //******************************************************************************************************************
    /** Interface DataBackend */
    void Flush();
    bool IsNull() const;
    size_type Count(std::string const& uri = "") const;

    std::shared_ptr<DataEntity> Get(std::string const& URI) const;
    std::shared_ptr<DataEntity> Get(id_type key) const;
    bool Set(std::string const& URI, std::shared_ptr<DataEntity> const&);
    bool Set(id_type key, std::shared_ptr<DataEntity> const&);
    bool Add(std::string const& URI, std::shared_ptr<DataEntity> const&);
    bool Add(id_type key, std::shared_ptr<DataEntity> const&);
    size_type Delete(std::string const& URI);
    size_type Delete(id_type key);

    size_type Accept(std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const&) const;
    size_type Accept(std::function<void(id_type, std::shared_ptr<DataEntity>)> const&) const;
    /** Interface End */
    //******************************************************************************************************************
    bool Set(DataTable const& other);
    bool Set(std::initializer_list<KeyValue> const& other);

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
        return Add(uri, make_data_entity(v));
    };
    template <typename U>
    bool Add(std::string const& uri, std::initializer_list<U> const& u) {
        return Add(uri, make_data_entity(u));
    };
    template <typename U>
    bool Add(std::string const& uri, std::initializer_list<std::initializer_list<U>> const& u) {
        return Add(uri, make_data_entity(u));
    };
    template <typename U>
    bool Add(U const& u) {
        return Add(make_data_entity(u));
    };
};

}  // namespace data
}  // namespace simpla

#endif  // SIMPLA_DATATABLE_H_
