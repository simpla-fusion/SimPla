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

   public:
    DataTable();
    DataTable(std::string const& uri, std::string const& param = "");
    DataTable(std::shared_ptr<DataBackend> const& p);
    DataTable(const DataTable&);
    DataTable(DataTable&&);
    ~DataTable() final;

    void swap(DataTable&);
    std::shared_ptr<DataEntity> Clone() const;
    //******************************************************************************************************************
    /** Interface DataEntity */
    std::ostream& Print(std::ostream& os, int indent = 0) const;
    bool isTable() const { return true; }
    std::type_info const& type() const { return typeid(DataTable); };

    DataBackend const* backend() const { return m_backend_.get(); }
    //******************************************************************************************************************
    /** Interface DataBackend */
    void Flush();
    bool isNull() const;
    size_type size() const;

    std::shared_ptr<DataEntity> Get(std::string const& uri) const;
    void Set(std::string const& uri, std::shared_ptr<DataEntity> const& p = nullptr);
    void Add(std::string const& uri, std::shared_ptr<DataEntity> const& p = nullptr);
    size_type Delete(std::string const& uri);
    size_type Accept(std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const&) const;

    /** Interface End */
    //******************************************************************************************************************

    bool has(std::string const& uri) const { return Get(uri) != nullptr; }

    DataTable& GetTable(std::string const& uri) {
        auto p = Get(uri);
        if (p != nullptr) {
            Set(uri);
            p = Get(uri);
        };
        ASSERT(p != nullptr);
        return p->cast_as<DataTable>();
    }
    DataTable const& GetTable(std::string const& uri) const {
        auto p = Get(uri);
        ASSERT(p != nullptr);
        return p->cast_as<DataTable>();
    }
    template <typename U>
    auto GetValue(std::string const& uri) const {
        return data_cast<U>(*Get(uri));
    }
    template <typename U>
    auto GetValue(std::string const& uri, U default_value) const {
        auto p = Get(uri);
        if (p == nullptr) { return default_value; }
        try {
            return data_cast<U>(*p);
        } catch (...) { return default_value; }
    }

    void SetValue(DataTable const& other);

    void SetValue(std::initializer_list<KeyValue> const& other);

    template <typename U>
    void SetValue(std::string const& uri, U const& v) {
        Set(uri, make_data_entity(v));
    };
    template <typename U>
    void SetValue(std::string const& uri, std::initializer_list<U> const& u) {
        Set(uri, make_data_entity(u));
    };
    template <typename U>
    void SetValue(std::string const& uri, std::initializer_list<std::initializer_list<U>> const& u) {
        Set(uri, make_data_entity(u));
    };
    template <typename U>
    void SetValue(std::string const& uri,
                  std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& u) {
        Set(uri, make_data_entity(u));
    };
    template <typename U>
    void AddValue(std::string const& uri, U const& v) {
        Add(uri, make_data_entity(v));
    };
    template <typename U>
    void AddValue(std::string const& uri, std::initializer_list<U> const& u) {
        Add(uri, make_data_entity(u));
    };
    template <typename U>
    void AddValue(std::string const& uri, std::initializer_list<std::initializer_list<U>> const& u) {
        Add(uri, make_data_entity(u));
    };
    template <typename U>
    void AddValue(U const& u) {
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
