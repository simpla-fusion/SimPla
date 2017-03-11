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
    std::shared_ptr<DataBackend> m_backend_;
    std::string m_base_uri_;

   public:
    DataTable();
    DataTable(std::string const& uri);
    DataTable(std::shared_ptr<DataBackend> const& p);
    DataTable(const DataTable&);
    DataTable(DataTable&&);
    ~DataTable() final;

    void swap(DataTable&);
    static std::shared_ptr<DataTable> Create(std::string const& scheme);

    std::shared_ptr<DataEntity> Clone() const;
    //******************************************************************************************************************
    /** Interface DataEntity */
    std::ostream& Print(std::ostream& os, int indent = 0) const;
    bool isTable() const { return true; }
    std::type_info const& type() const { return typeid(DataTable); };

    DataBackend const* backend() const { return m_backend_.get(); }
    std::string const& base_uri() const { return m_base_uri_; }
    //******************************************************************************************************************
    /** Interface DataBackend */
    void Flush();
    bool isNull() const;
    size_type size() const;

    std::shared_ptr<DataTable> AddTable(std::string const&);
    std::shared_ptr<DataEntity> Get(std::string const& path) const;
    void Set(std::string const& path, std::shared_ptr<DataEntity> const&);
    void Add(std::string const& path, std::shared_ptr<DataEntity> const&);
    size_type Delete(std::string const& path);
    size_type Accept(std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const&) const;
    /** Interface End */
    //******************************************************************************************************************
    void Set(DataTable const& other);
    void Set(std::initializer_list<KeyValue> const& other);

    template <typename U>
    void Set(std::string const& uri, U const& v) {
        Set(uri, make_data_entity(v));
    };
    template <typename U>
    bool Set(std::string const& uri, std::initializer_list<U> const& u) {
        Set(uri, make_data_entity(u));
    };
    template <typename U>
    void Set(std::string const& uri, std::initializer_list<std::initializer_list<U>> const& u) {
        Set(uri, make_data_entity(u));
    };
    template <typename U>
    void Set(std::string const& uri, std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& u) {
        Set(uri, make_data_entity(u));
    };
    template <typename U>
    void Add(std::string const& uri, U const& v) {
        Add(uri, make_data_entity(v));
    };
    template <typename U>
    void Add(std::string const& uri, std::initializer_list<U> const& u) {
        Add(uri, make_data_entity(u));
    };
    template <typename U>
    void Add(std::string const& uri, std::initializer_list<std::initializer_list<U>> const& u) {
        Add(uri, make_data_entity(u));
    };
    template <typename U>
    void Add(U const& u) {
        Add("", make_data_entity({u}));
    };

    template <typename U>
    void AddArray() {
        Add(make_data_entity(std::initializer_list<U>{}));
    };
};

}  // namespace data
}  // namespace simpla

#endif  // SIMPLA_DATATABLE_H_
