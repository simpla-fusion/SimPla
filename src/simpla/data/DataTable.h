//
// Created by salmon on 16-10-6.
//

#ifndef SIMPLA_DATATABLE_H_
#define SIMPLA_DATATABLE_H_

#include <memory>
#include "DataArray.h"
#include "DataEntity.h"
#include "DataTraits.h"
#include "simpla/SIMPLA_config.h"
#include "simpla/utilities/ObjectHead.h"
namespace simpla {
namespace data {

class DataBackend;

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
    ~DataTable() override = default;

    explicit DataTable(std::shared_ptr<DataBackend> const& p);
    DataTable(std::string const& uri, std::string const& param = "");

    DataTable(const DataTable&);
    DataTable(DataTable&&) noexcept;
    void swap(DataTable& other);

    //******************************************************************************************************************
    /** Interface DataEntity */

    std::ostream& Serialize(std::ostream& os, int indent) const override;
    std::istream& Deserialize(std::istream& is) override;

    std::shared_ptr<DataEntity> Duplicate() const override {
        return std::dynamic_pointer_cast<DataEntity>(std::make_shared<this_type>(*this));
    }  //******************************************************************************************************************
    /** Interface DataBackend */

    std::shared_ptr<DataBackend> backend() const { return m_backend_; }

    void Flush();
    bool isNull() const override;
    size_type size() const;

    void Set(std::string const& uri, const std::shared_ptr<DataEntity>& src);
    void Add(std::string const& uri, const std::shared_ptr<DataEntity>& p);
    std::shared_ptr<DataEntity> Get(std::string const& uri);
    std::shared_ptr<DataEntity> Get(std::string const& uri) const;
    size_type Delete(std::string const& uri);

    void SetTable(DataTable const& other);
    DataTable& GetTable(std::string const& uri);
    const DataTable& GetTable(std::string const& uri) const;

    void Set(std::string const& uri, bool flag = true) { Set(uri, make_data_entity(flag)); };

    //    void Assign(){};
    //    void Assign(std::string const& uri) { SetValue(uri, true); };
    //    void Assign(DataTable const& v) { Set(v); };
    //
    //    template <typename First, typename Second, typename... Others>
    //    void Assign(First&& first, Second&& second, Others&&... args) {
    //        Assign(std::forward<First>(first));
    //        Assign(std::forward<Second>(second), std::forward<Others>(args)...);
    //    }

    size_type Foreach(std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const&) const;

    /** Interface DataBackend End */
    //******************************************************************************************************************
    bool has(std::string const& uri) const { return Get(uri) != nullptr; }
    bool isTable(std::string const& uri) const { return std::dynamic_pointer_cast<DataTable>(Get(uri)) != nullptr; }
    bool isArray(std::string const& uri) const { return std::dynamic_pointer_cast<DataArray>(Get(uri)) != nullptr; }

    template <typename U>
    bool Check(std::string const& key, U const& u = true) const {
        auto p = std::dynamic_pointer_cast<DataEntityWrapper<U>>(Get(key));
        return (p != nullptr) && p->value() == u;
    }
    template <typename U>
    bool CheckType(std::string const& uri) const {
        return std::dynamic_pointer_cast<DataEntityWrapper<U>>(Get(uri)) != nullptr;
    }
    bool Check(std::string const& key) const { return Check(key, true); }

    template <typename U>
    U GetValue(std::string const& uri) const {
        auto res = std::dynamic_pointer_cast<DataEntityWrapper<U>>(Get(uri));
        if (res == nullptr) { OUT_OF_RANGE << "Can not find entity [" << uri << "]" << std::endl; }
        return res->value();
    }

    template <typename U>
    U GetValue(std::string const& uri, U const& default_value) const {
        auto res = std::dynamic_pointer_cast<DataEntityWrapper<U>>(Get(uri));
        return res == nullptr ? default_value : res->value();
    }

    void SetValue(std::string const& uri, DataTable const& v) { GetTable(uri).SetTable(v); };

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
};

struct KeyHolder {
    KeyHolder(char const* c) : m_key_(c) {}
    KeyHolder(std::string const& s) : m_key_(s) {}
    ~KeyHolder() = default;

    template <typename U>
    DataTable operator=(U const& v) {
        DataTable res;
        res.SetValue(m_key_, v);
        return std::move(res);
    }

    operator DataTable() const {
        DataTable res;
        res.SetValue(m_key_, true);
        return std::move(res);
    }
    std::string m_key_;
};
inline KeyHolder operator"" _(const char* c, std::size_t n) { return KeyHolder{std::string(c)}; }

// template <typename... Others>
// std::shared_ptr<DataEntity> make_data_entity(KeyValue const& first, Others&&... others) {
//    auto res = std::make_shared<DataTable>();
//    res->SetValue(first, std::forward<Others>(others)...);
//    return res;
//}

}  // namespace data
}  // namespace simpla

#endif  // SIMPLA_DATATABLE_H_
