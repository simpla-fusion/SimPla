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

class KeyValue : public std::pair<std::string, std::shared_ptr<DataEntity>> {
    typedef std::pair<std::string, std::shared_ptr<DataEntity>> base_type;

   public:
    explicit KeyValue(std::string const& k, std::shared_ptr<DataEntity> const& p = nullptr) : base_type(k, p) {}
    KeyValue(KeyValue const& other) : base_type(other) {}
    KeyValue(KeyValue&& other) : base_type(other) {}
    ~KeyValue() = default;

    KeyValue& operator=(KeyValue const& other) {
        base_type::operator=(other);
        return *this;
    }

    template <typename U>
    KeyValue& operator=(U const& u) {
        second = DataEntity::New(u);
        return *this;
    }
    template <typename U>
    KeyValue& operator=(std::initializer_list<U> const& u) {
        second = DataEntity::New(u);
        return *this;
    }
    template <typename U>
    KeyValue& operator=(std::initializer_list<std::initializer_list<U>> const& u) {
        second = DataEntity::New(u);
        return *this;
    }
    template <typename U>
    KeyValue& operator=(std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& u) {
        second = DataEntity::New(u);
        return *this;
    }
};

inline KeyValue operator"" _(const char* c, std::size_t n) { return KeyValue{std::string(c), DataEntity::New(true)}; }

class DataNode : public std::enable_shared_from_this<DataNode> {
    SP_OBJECT_BASE(DataNode);

   protected:
    DataNode();

   public:
    virtual ~DataNode();
    SP_DEFAULT_CONSTRUCT(DataNode);

    static std::shared_ptr<DataNode> New(std::string const& k = "");

    virtual std::shared_ptr<DataNode> Duplicate() const { return DataNode::New(); }

    enum { RECURSIVE = 0b01, NEW_IF_NOT_EXIST = 0b010, ADD_IF_EXIST = 0b100, ONLY_TABLE = 0b1000 };

    /** @addtogroup{ Interface */
    virtual int Flush() { return 0; }
    virtual bool isNull() const { return true; }
    virtual bool isArray() const { return false; }
    virtual bool isTable() const { return false; }
    virtual bool isEntity() const { return false; }

    virtual size_type GetNumberOfChildren() const { return 0; }

    virtual std::shared_ptr<DataNode> Root() { return DataNode::New(); }
    virtual std::shared_ptr<DataNode> Parent() const { return DataNode::New(); }
    virtual std::shared_ptr<DataNode> FirstChild() const { return nullptr; }
    virtual std::shared_ptr<DataNode> Next() const { return nullptr; }

    virtual std::shared_ptr<DataNode> GetNode(std::string const& uri = "", int flag = 0) { return New(); }
    virtual std::shared_ptr<DataNode> GetNode(std::string const& uri = "", int flag = 0) const { return New(); }
    virtual std::shared_ptr<DataNode> GetNode(index_type s, int flag = 0) { return GetNode(std::to_string(s), flag); };
    virtual std::shared_ptr<DataNode> GetNode(index_type s, int flag = 0) const {
        return GetNode(std::to_string(s), flag);
    };

    virtual int DeleteNode(std::string const& s, int flag = 0) { return 0; }
    virtual int DeleteNode(index_type s, int flag = 0) { return DeleteNode(std::to_string(s), flag); };

    virtual std::string GetKey() const { return "KEY:"; }
    virtual std::shared_ptr<DataEntity> GetEntity() { return DataEntity::New(); }
    virtual std::shared_ptr<DataEntity> GetEntity() const { return DataEntity::New(); }
    virtual int SetEntity(std::shared_ptr<DataEntity> const& v) { return 0; }
    /** @} */

    template <typename U>
    int SetValue(U const& u) {
        return SetEntity(std::dynamic_pointer_cast<DataEntity>(DataEntity::New(std::forward<U>(u))));
    }
    template <typename U>
    int AddValue(U&& u) {
        return GetNode(".", RECURSIVE | ADD_IF_EXIST | NEW_IF_NOT_EXIST)->SetEntity(std::forward<U>(u));
    }
    template <typename U>
    U as() const {
        return GetEntity()->as<U>();
    }

    template <typename U>
    U as(U const& default_value) const {
        return GetEntity()->as<U>(default_value);
    }
    template <typename U, typename URL>
    U GetValue(URL const& url) const {
        return GetNode(url, RECURSIVE)->template as<U>();
    }
    template <typename U, typename URL>
    U GetValue(URL const& url, U const& default_value) const {
        return GetNode(url, RECURSIVE)->template as<U>(default_value);
    }
    template <typename U, typename URL>
    int SetValue(URL const& url, U&& u) {
        return GetNode(url, RECURSIVE | NEW_IF_NOT_EXIST)->SetEntity(DataEntity::New(std::forward<U>(u)));
    }
    template <typename U, typename URL>
    int SetValue(URL const& url, std::initializer_list<U> const& u) {
        return GetNode(url, RECURSIVE | NEW_IF_NOT_EXIST)->SetEntity(DataEntity::New(u));
    }
    template <typename U, typename URL>
    int SetValue(URL const& url, std::initializer_list<std::initializer_list<U>> const& u) {
        return GetNode(url, RECURSIVE | NEW_IF_NOT_EXIST)->SetEntity(DataEntity::New(u));
    }
    template <typename U, typename URL>
    int SetValue(URL const& url, std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& u) {
        return GetNode(url, RECURSIVE | NEW_IF_NOT_EXIST)->SetEntity(DataEntity::New(u));
    }

    template <typename U, typename URL>
    int AddValue(URL const& url, U&& u) {
        return GetNode(url, RECURSIVE | NEW_IF_NOT_EXIST | ADD_IF_EXIST)
            ->SetEntity(DataEntity::New(std::forward<U>(u)));
    };
    template <typename U, typename URL>
    int AddValue(URL const& url, std::initializer_list<U> const& u) {
        return GetNode(url, RECURSIVE | NEW_IF_NOT_EXIST | ADD_IF_EXIST)->SetEntity(DataEntity::New(u));
    }
    template <typename U, typename URL>
    int AddValue(URL const& url, std::initializer_list<std::initializer_list<U>> const& u,
                 int flag = RECURSIVE | NEW_IF_NOT_EXIST) {
        return GetNode(url, RECURSIVE | NEW_IF_NOT_EXIST | ADD_IF_EXIST)->SetEntity(DataEntity::New(u));
    }
    template <typename U, typename URL>
    int AddValue(URL const& url, std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& u,
                 int flag = RECURSIVE | NEW_IF_NOT_EXIST) {
        return GetNode(url, RECURSIVE | NEW_IF_NOT_EXIST | ADD_IF_EXIST)->SetEntity(DataEntity::New(u));
    }

    template <typename U, typename URL>
    bool Check(URL const& url, U const& u) const {
        return GetNode(url, RECURSIVE)->GetEntity()->equal(u);
    }
    bool Check(std::string const& uri) const { return Check(uri, true); }

    int SetValue(KeyValue const& kv) { return SetValue(kv.first, kv.second); }

    template <typename... Others>
    int SetValue(KeyValue const& kv, Others&&... others) {
        return SetValue(kv) + SetEntity(std::forward<Others>(others)...);
    }
    int SetValue(std::initializer_list<KeyValue> const& u) {
        int count = 0;
        for (auto const& item : u) { count += static_cast<int>(SetValue(item)); }
        return count;
    }
    int SetValue(std::initializer_list<std::initializer_list<KeyValue>> const& u) {
        int count = 0;
        for (auto const& item : u) { count += static_cast<int>(SetValue(item)); }
        return count;
    }
    int SetValue(std::initializer_list<std::initializer_list<std::initializer_list<KeyValue>>> const& u) {
        int count = 0;
        for (auto const& item : u) { count += static_cast<int>(SetValue(item)); }
        return count;
    }

    /** @} */
};
std::pair<std::shared_ptr<DataNode>, std::string> RecursiveFindNode(std::shared_ptr<DataNode> const& d,
                                                                    std::string const& uri, int flag = 0);
// std::shared_ptr<const DataNode> RecursiveFindNode(std::shared_ptr<const DataNode> const& d, std::string const& uri,
//                                                  int flag = 0);
std::ostream& operator<<(std::ostream&, DataNode const&);
}  // namespace data
}  // namespace simpla

#endif  // SIMPLA_DATAENTRY_H
