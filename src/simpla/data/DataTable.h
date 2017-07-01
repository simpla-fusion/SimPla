//
// Created by salmon on 16-10-6.
//

#ifndef SIMPLA_DATATABLE_H_
#define SIMPLA_DATATABLE_H_

#include <simpla/SIMPLA_config.h>
#include <simpla/engine/SPObject.h>
#include <memory>
#include "DataArray.h"
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
 * in HDF5,  all node/table are DataEntity.
 * @design_pattern
 *  - Proxy for DataBackend
 */
class DataTable : public DataEntity {
    SP_OBJECT_HEAD(DataTable, DataEntity);
    std::shared_ptr<DataBackend> m_backend_;

   public:
    DataTable();
    DataTable(const DataTable&);
    DataTable(DataTable&&) noexcept ;
    virtual ~DataTable();

    DataTable& operator=(const DataTable& other) {
        this_type(other).swap(*this);
        return *this;
    }
    DataTable& operator=(DataTable&& other) noexcept {
        this_type(other).swap(*this);
        return *this;
    }

    DataTable(std::string const& uri, std::string const& param = "");
    explicit DataTable(std::shared_ptr<DataBackend> const& p);

    DataTable(std::initializer_list<KeyValue> const& l);

    template <typename... Others>
    explicit DataTable(KeyValue const& v, Others&&... others) : DataTable() {
        SetValue(v, std::forward<Others>(others)...);
    }
    void swap(DataTable&);
    //******************************************************************************************************************
    /** Interface DataEntity */

    std::ostream& Serialize(std::ostream& os, int indent) const override;
    std::istream& Deserialize(std::istream& is) override;

    bool isTable() const override { return true; }
    std::type_info const& value_type_info() const override { return typeid(DataTable); };
    std::shared_ptr<DataEntity> Duplicate() const override;
    //******************************************************************************************************************
    /** Interface DataBackend */

    std::shared_ptr<DataBackend> backend() const { return m_backend_; }

    void Flush();
    bool isNull() const override;
    size_type size() const;

    std::shared_ptr<DataEntity> Get(std::string const& uri) const;

    void Set(std::string const& uri, std::shared_ptr<DataEntity> const& p = nullptr, bool overwrite = true);
    void Add(std::string const& uri, std::shared_ptr<DataEntity> const& p = nullptr);
    void Delete(std::string const& uri);
    size_type Foreach(std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const&) const;

    /** Interface DataBackend End */
    //******************************************************************************************************************
    bool has(std::string const& uri) const { return Get(uri) != nullptr; }

    template <typename U>
    bool Check(std::string const& key, U const& u = true) const {
        auto p = Get(key);
        return (p != nullptr) && (p->value_type_info() == typeid(U)) && (DataCastTraits<U>::Get(p) == u);
    }
    bool Check(std::string const& key) const { return Check(key, true); }

    template <typename U>
    bool CheckType(std::string const& uri) const {
        auto r = Get(uri);
        return r != nullptr && r->value_type_info() == typeid(U);
    }

    void Link(std::shared_ptr<DataEntity> const& other);
    DataTable& Link(std::string const& uri, DataTable const& other);
    DataTable& Link(std::string const& uri, std::shared_ptr<DataEntity> const& p);

    void Set(std::shared_ptr<DataTable> const& other, bool overwrite = true);

    void Set(DataTable const& other, bool overwrite = true);
    void Set(std::string const& uri, DataEntity const& p, bool overwrite = true);
    void Add(std::string const& uri, DataEntity const& p);

    std::shared_ptr<DataTable> GetTable(std::string const& uri) const;

    template <typename U>
    U GetValue(std::string const& uri) const {
        return DataCastTraits<U>::Get(Get(uri));
    }

    template <typename U>
    U GetValue(std::string const& uri, U const& default_value) const {
        return DataCastTraits<U>::Get(Get(uri), default_value);
    }

    //    template <typename U>
    //    U GetValue(std::string const& uri, U const& default_value) {
    //        Unpack(uri, make_data_entity(default_value), false);
    //        return data_cast<U>(*Pack(uri));
    //    }

    template <typename U>
    DataTable& operator=(U const& u) {
        SetValue(u);
        return *this;
    }
    template <typename U>
    DataTable& operator=(std::initializer_list<U> const& u) {
        SetValue(u);
        return *this;
    }
    //    template <typename U>
    //    void SetValue(std::pair<std::string, U> const& item) {
    //        SetValue(item.first, item.second);
    //    }
    //    template <typename U>
    //    void SetValue(std::initializer_list<std::pair<std::string, U>> const& other) {
    //        for (auto const& item : other) { SetValue(item.first, item.second); }
    //    }
    void SetValue(){};
    void SetValue(KeyValue const& other);
    void SetValue(std::initializer_list<KeyValue> const& other);
    template <typename... Others>
    void SetValue(KeyValue const& first, Others&&... others) {
        SetValue(first);
        SetValue(std::forward<Others>(others)...);
    };

    template <typename U>
    void SetValue(std::string const& uri, U const& v, bool overwrite = false) {
        Set(uri, make_data_entity(v), overwrite);
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
template <typename... Others>
std::shared_ptr<DataEntity> make_data_entity(KeyValue const& first, Others&&... others) {
    auto res = std::make_shared<DataTable>();
    res->SetValue(first, std::forward<Others>(others)...);
    return res;
}

}  // namespace data
}  // namespace simpla

#endif  // SIMPLA_DATATABLE_H_
