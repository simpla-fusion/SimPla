//
// Created by salmon on 17-8-18.
//

#ifndef SIMPLA_DATAENTRY_H
#define SIMPLA_DATAENTRY_H

#include "simpla/SIMPLA_config.h"

#include <memory>
#include "DataEntity.h"

#include "simpla/utilities/Log.h"
#include "simpla/utilities/ObjectHead.h"
namespace simpla {
namespace data {
class DataEntity;
class KeyValue;
class DataNode : public std::enable_shared_from_this<DataNode> {
    SP_OBJECT_BASE(DataNode);

   protected:
    DataNode();

   public:
    virtual ~DataNode();
    SP_DEFAULT_CONSTRUCT(DataNode);

    static std::shared_ptr<DataNode> New(std::string const& k = "");

    /** @addtogroup{ capacity */
    virtual bool isNull() { return true; }
    /** @} */
    /** @addtogroup{ access */
    virtual std::shared_ptr<DataNode> Root() { return GetNode("/", RECUSIVE); }
    virtual std::shared_ptr<DataNode> Parent() const { return GetNode("..", RECUSIVE); }

    virtual std::shared_ptr<DataNode> FirstChild() const { return nullptr; }
    virtual std::shared_ptr<DataNode> Next() const { return nullptr; }

    enum { SUB_NODE, RECUSIVE = 0b01, NEW_IF_NOT_EXIST = 0b010, ADD_IF_EXIST = 0b100 };

    virtual std::shared_ptr<DataNode> GetNode(std::string const& uri, int flag) { return nullptr; };
    virtual std::shared_ptr<DataNode> GetNode(std::string const& uri, int flag) const { return nullptr; };

    /** @} */

    /** @addtogroup{  Value */
    virtual std::shared_ptr<DataEntity> GetValue() { return nullptr; }
    virtual std::shared_ptr<DataEntity> GetValue() const { return nullptr; }
    virtual int SetValue(std::shared_ptr<DataEntity> const& v) { return 0; }

    virtual int SetValueByName(std::string const& uri, std::shared_ptr<DataEntity> const& v) {
        auto p = GetNode(uri, RECUSIVE | NEW_IF_NOT_EXIST);
        return p == nullptr ? 0 : p->SetValue(v);
    }
    virtual int SetValueByIndex(index_type idx, std::shared_ptr<DataEntity> const& v) {
        return SetValueByName(std::to_string(idx), v);
    }

    virtual size_type GetNumberOfChildren() const { return 0; }

    virtual std::shared_ptr<DataEntity> GetValueByIndex(index_type idx) const {
        return GetValueByName(std::to_string(idx), SUB_NODE);
    }
    virtual std::shared_ptr<DataEntity> GetValueByName(std::string const& uri, int flag) const {
        auto p = GetNode(uri, flag);
        return p == nullptr ? nullptr : p->GetValue();
    }

    template <typename V>
    V GetValue(std::string const& k, int flag = RECUSIVE) const {
        auto p = GetNode(k, RECUSIVE);
        if (p == nullptr) { OUT_OF_RANGE; }
        return p->GetValue()->as<V>();
    }

    template <typename V>
    V GetValue(std::string const& k, V const& default_value, int flag = RECUSIVE) const {
        auto p = GetNode(k, RECUSIVE);
        return p == nullptr ? default_value : p->GetValue()->as<V>();
    }

    /** Interface DataBackend End */

    template <typename U>
    bool Check(std::string const& uri, U const& u) const {
        auto const& p = GetNode(uri, RECUSIVE);
        return p != nullptr && p->Check(u);
    }
    bool Check(std::string const& uri) const { return Check(uri, true); }

    template <typename U>
    U GetValue(std::string const& uri) const {
        auto res = GetNode(uri, RECUSIVE);
        if (res == nullptr) { OUT_OF_RANGE << "Can not find entity [" << uri << "]" << std::endl; }
        return res->as<U>();
    }

    template <typename U>
    U GetValue(std::string const& uri, U const& default_value) const {
        auto res = GetNode(uri, RECUSIVE);
        return res == nullptr ? default_value : res->as<U>();
    }

    void SetValue(KeyValue const& kv) { GetNode(kv.first, RECUSIVE | NEW_IF_NOT_EXIST)->SetValue(kv.second); }
    template <typename... Others>
    void SetValue(KeyValue const& kv, Others&&... others) {
        GetNode(kv.first, RECUSIVE | NEW_IF_NOT_EXIST)->SetValue(kv.second);
        SetValue(std::forward<Others>(others)...);
    }
    void SetValue(std::initializer_list<KeyValue> const& u) {
        for (auto const& item : u) { SetValue(item); }
    }

    template <typename U>
    void SetValue(std::string const& uri, U const& v) {
        GetNode(uri, RECUSIVE | NEW_IF_NOT_EXIST)->SetValue(make_data_entity(v));
    };

    template <typename U>
    void AddValue(std::string const& uri, U const& v) {
        GetNode(uri, RECUSIVE | NEW_IF_NOT_EXIST | ADD_IF_EXIST)->SetValue(make_data_entity(v));
    };

    /** @} */
};

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
std::ostream& operator<<(std::ostream, DataNode const&);
}  // namespace data
}  // namespace simpla

#endif  // SIMPLA_DATAENTRY_H
