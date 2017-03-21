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
 * in HDF5,  all node/table are DataEntity.
 * @design_pattern
 *  - Proxy for DataBackend
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
    //******************************************************************************************************************
    /** Interface DataEntity */
    std::ostream& Print(std::ostream& os, int indent = 0) const;
    bool isTable() const { return true; }
    std::type_info const& value_type_info() const { return typeid(DataTable); };
    std::shared_ptr<DataEntity> Duplicate() const;
    //******************************************************************************************************************
    /** Interface DataBackend */

    std::shared_ptr<DataBackend> backend() const { return m_backend_; }

    void Flush();
    bool isNull() const;
    size_type size() const;

    std::shared_ptr<DataEntity> Get(std::string const& uri) const;
    int Set(std::string const& uri, std::shared_ptr<DataEntity> const& p = nullptr, bool overwrite = true);
    int Add(std::string const& uri, std::shared_ptr<DataEntity> const& p = nullptr);
    size_type Delete(std::string const& uri);
    size_type Foreach(std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const&) const;

    /** Interface DataBackend End */
    //******************************************************************************************************************
    bool has(std::string const& uri) const { return Get(uri) != nullptr; }

    void Link(std::shared_ptr<DataEntity> const& other);
    DataTable& Link(std::string const& uri, DataTable const& other);
    DataTable& Link(std::string const& uri, std::shared_ptr<DataEntity> const& p);

    void Set(DataTable const& other, bool overwrite = true);
    int Set(std::string const& uri, DataEntity const& p, bool overwrite = true);
    int Add(std::string const& uri, DataEntity const& p);

    std::shared_ptr<DataTable> GetTable(std::string const& uri) const;

    template <typename U>
    U GetValue(std::string const& uri) const {
        return data_cast<U>(*Get(uri));
    }

    template <typename U>
    U GetValue(std::string const& uri, U const& default_value) const {
        auto p = Get(uri);
        if (p == nullptr || p->isNull() || p->value_type_info() != typeid(U)) {
            return default_value;
        } else {
            return data_cast<U>(*p);
        }
    }

//    template <typename U>
//    U GetValue(std::string const& uri, U const& default_value) {
//        Set(uri, make_data_entity(default_value), false);
//        return data_cast<U>(*Get(uri));
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
