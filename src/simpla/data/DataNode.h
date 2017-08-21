//
// Created by salmon on 17-8-18.
//

#ifndef SIMPLA_DATAENTRY_H
#define SIMPLA_DATAENTRY_H

#include "simpla/SIMPLA_config.h"

#include <memory>
#include "DataLight.h"
#include "DataUtilities.h"
#include "simpla/utilities/Log.h"
#include "simpla/utilities/ObjectHead.h"
namespace simpla {
namespace data {

class DataLight;

class DataNode : public std::enable_shared_from_this<DataNode> {
    SP_OBJECT_BASE(DataNode);

   protected:
    DataNode();

   public:
    virtual ~DataNode();
    DataNode(DataNode const& other) = delete;
    DataNode(DataNode&& other) = delete;

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

    virtual std::shared_ptr<DataNode> GetNode(std::string const& uri = "", int flag = RECURSIVE | NEW_IF_NOT_EXIST) {
        return New();
    }
    virtual std::shared_ptr<DataNode> GetNode(std::string const& uri = "", int flag = RECURSIVE) const { return New(); }
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
    DataNode& operator[](std::string const& s) { return *GetNode(s, NEW_IF_NOT_EXIST); }
    template <typename U>
    DataNode& operator=(U const& u) {
        SetValue<U>(u);
        return *this;
    }
    template <typename U>
    DataNode& operator=(std::initializer_list<U> const& u) {
        SetEntity(make_data_entity(u));
        return *this;
    }
    template <typename U>
    DataNode& operator=(std::initializer_list<std::initializer_list<U>> const& u) {
        SetEntity(make_data_entity(u));
        return *this;
    }
    template <typename U>
    DataNode& operator=(std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& u) {
        SetEntity(make_data_entity(u));
        return *this;
    }

    DataNode& operator=(KeyValue const& kv) {
        SetValue(kv);
        return *this;
    }
    DataNode& operator=(std::initializer_list<KeyValue> const& u) {
        SetValue(u);
        return *this;
    }
    DataNode& operator=(std::initializer_list<std::initializer_list<KeyValue>> const& u) {
        SetValue(u);
        return *this;
    }
    DataNode& operator=(std::initializer_list<std::initializer_list<std::initializer_list<KeyValue>>> const& u) {
        SetValue(u);
        return *this;
    }

    int SetValue(std::shared_ptr<DataEntity> const& u) { return SetEntity(u); }

    template <typename U, typename... Args>
    int SetValue(Args&&... args) {
        return SetEntity(DataLightT<U>::New(std::forward<Args>(args)...));
    }

    int SetValue(KeyValue const& kv) { return GetNode(kv.first, RECURSIVE | NEW_IF_NOT_EXIST)->SetValue(kv.second); }

    template <typename... Others>
    int SetValue(KeyValue const& kv, Others&&... others) {
        return SetValue(kv) + SetEntity(std::forward<Others>(others)...);
    }
    int SetValue(std::initializer_list<KeyValue> const& u) {
        int count = 0;
        for (auto const& item : u) { count += (SetValue(item)); }
        return count;
    }
    int SetValue(std::initializer_list<std::initializer_list<KeyValue>> const& u) {
        int count = 0;
        for (auto const& item : u) { count += (SetValue(item)); }
        return count;
    }
    int SetValue(std::initializer_list<std::initializer_list<std::initializer_list<KeyValue>>> const& u) {
        int count = 0;
        for (auto const& item : u) { count += (SetValue(item)); }
        return count;
    }
    template <typename URL, typename U>
    bool Check(URL const& url, U const& u) const {
        return GetNode(url, RECURSIVE)->GetEntity()->equal(u);
    }
    bool Check(std::string const& uri) const { return Check(uri, true); }

    template <typename U>
    int AddValue(U const& u) {
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
