//
// Created by salmon on 16-10-6.
//

#ifndef SIMPLA_DATATABLE_H_
#define SIMPLA_DATATABLE_H_
#include "simpla/SIMPLA_config.h"

#include <memory>
#include "DataArray.h"
#include "DataEntity.h"
#include "DataTraits.h"
#include "simpla/utilities/ObjectHead.h"
namespace simpla {
namespace data {

class DataBase;
class KeyValue;

class KeyValue : public std::pair<std::string, std::shared_ptr<DataEntity>> {
    typedef std::pair<std::string, std::shared_ptr<DataEntity>> base_type;

   public:
    explicit KeyValue(std::string const& k, std::shared_ptr<DataEntity> const& p = nullptr) : base_type(k, p) {}
    KeyValue(KeyValue const& other) : base_type(other) {}
    KeyValue(KeyValue&& other) : base_type(other) {}
    ~KeyValue() = default;

    KeyValue& operator=(KeyValue const& other) {
        //        base_type::operator=(other);
        return *this;
    }

    template <typename U>
    KeyValue& operator=(U const& u) {
        second = make_data_entity(u);
        return *this;
    }
    template <typename U>
    KeyValue& operator=(std::initializer_list<U> const& u) {
        second = make_data_entity(u);
        return *this;
    }
    template <typename U>
    KeyValue& operator=(std::initializer_list<std::initializer_list<U>> const& u) {
        second = make_data_entity(u);
        return *this;
    }
    template <typename U>
    KeyValue& operator=(std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& u) {
        second = make_data_entity(u);
        return *this;
    }
};

inline KeyValue operator"" _(const char* c, std::size_t n) { return KeyValue{std::string(c), make_data_entity(true)}; }
/** @ingroup data */
/**
 * @brief  a @ref DataEntity tree, a key-value table of @ref DataEntity, which is similar as Group
 * in HDF5,  all node/table are DataEntity.
 * @design_pattern
 *  - Proxy for DataBackend
 *
 *  PUT and POST are both unsafe methods. However, PUT is idempotent, while POST is not.
 *
 *  HTTP/1.1 SPEC
 *  @quota
 *   The POST method is used to request that the origin server accept the entity enclosed in
 *   the request as a new subordinate of the resource identified by the Request-URI in the Request-Line
 *
 *  @quota
 *  The PUT method requests that the enclosed entity be stored under the supplied Request-URI.
 *  If the Request-URI refers to an already existing resource, the enclosed entity SHOULD be considered as a
 *  modified version of the one residing on the origin server. If the Request-URI does not point to an existing
 *  resource, and that URI is capable of being defined as a new resource by the requesting user agent, the origin
 *  server can create the resource with that URI."
 *
 */

class DataTable : public DataNode {
    SP_OBJECT_HEAD(DataTable, DataNode);
    struct pimpl_s;
    pimpl_s* m_pimpl_;

   protected:
    DataTable();

   public:
    ~DataTable() override;
    SP_DEFAULT_CONSTRUCT(DataTable);

    template <typename... Args>
    static std::shared_ptr<DataTable> New(Args&&... args) {
        return std::shared_ptr<DataTable>(new DataTable(std::forward<Args>(args)...));
    }
    size_type GetNumberOfChildren() const override;
    std::shared_ptr<DataNode> Child() const override;

    std::shared_ptr<DataNode> GetNodeByName(std::string const& s) const override;
    int SetNodeByName(std::string const& k, std::shared_ptr<DataNode> v) override;
    std::shared_ptr<DataNode> GetNodeByIndex(index_type s) const override { return GetNodeByName(std::to_string(s)); }
    int SetNodeByIndex(index_type idx, std::shared_ptr<DataNode> const& v) override {
        return SetNodeByName(std::to_string(idx), v);
    }
    int AddValue(std::shared_ptr<DataEntity> const& v) override {
        return SetNodeByIndex(GetNumberOfChildren(), DataNodeWithKey::New(v));
    }
    //******************************************************************************************************************
    /** Interface DataEntity */

    bool has(std::string const& uri) const { return Get(uri) != nullptr; }

    bool isNull() const;
    size_type Count() const;

    std::shared_ptr<DataEntity>& Get(std::string const& uri);
    std::shared_ptr<DataEntity> const& Get(std::string const& uri) const;
    int Set(std::string const& uri, const std::shared_ptr<DataEntity>& v);
    int Add(std::string const& uri, const std::shared_ptr<DataEntity>& v);
    int Delete(std::string const& uri);
    int Set(const std::shared_ptr<DataTable>& v);

    /** Interface DataBackend End */

    template <typename U>
    bool Check(std::string const& uri, U const& u) const {
        auto const& p = Get(uri);
        return p->Check(u);
    }
    bool Check(std::string const& uri) const { return Check(uri, true); }

    template <typename U>
    U GetValue(std::string const& uri) const {
        auto res = std::dynamic_pointer_cast<const DataLight>(Get(uri));
        if (res == nullptr) { OUT_OF_RANGE << "Can not find entity [" << uri << "]" << std::endl; }
        return res->as<U>();
    }

    template <typename U>
    U GetValue(std::string const& uri, U const& default_value) const {
        auto res = std::dynamic_pointer_cast<const DataLight>(Get(uri));
        return res == nullptr ? default_value : res->as<U>();
    }

    void SetValue(KeyValue const& kv) { Set(kv.first, kv.second); }
    template <typename... Others>
    void SetValue(KeyValue const& kv, Others&&... others) {
        Set(kv.first, kv.second);
        SetValue(std::forward<Others>(others)...);
    }
    void SetValue(std::initializer_list<KeyValue> const& u) {
        for (auto const& item : u) { SetValue(item); }
    }

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

}  // namespace data

// template <typename U, typename... Args>
// std::shared_ptr<U> CreateObject(data::DataEntity const* dataEntity, Args&&... args) {
//    std::shared_ptr<U> res = nullptr;
//    if (dynamic_cast<data::DataLight<std::string> const*>(dataEntity) != nullptr) {
//        res = U::Create(dynamic_cast<data::DataLight<std::string> const*>(dataEntity)->value(),
//                        std::forward<Args>(args)...);
//    } else if (dynamic_cast<data::DataTable const*>(dataEntity) != nullptr) {
//        auto const* db = dynamic_cast<data::DataTable const*>(dataEntity);
//        res = U::Create(db->GetValue<std::string>("Type", ""), std::forward<Args>(args)...);
//        res->Deserialize(*db);
//    } else {
//        res = U::Create("", std::forward<Args>(args)...);
//    }
//    return res;
//};

}  // namespace simpla

#endif  // SIMPLA_DATATABLE_H_
