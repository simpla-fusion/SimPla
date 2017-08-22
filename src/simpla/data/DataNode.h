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

class DataNode;
typedef std::shared_ptr<DataNode> DataNodeP;
class DataNode : public std::enable_shared_from_this<DataNode> {
    SP_OBJECT_BASE(DataNode);

   protected:
    DataNode();

   public:
    virtual ~DataNode();
    DataNode(DataNode const& other) = delete;
    DataNode(DataNode&& other) = delete;

    static std::shared_ptr<DataNode> New(std::string const& uri = "");

    enum { RECURSIVE = 0b01, NEW_IF_NOT_EXIST = 0b010, ADD_IF_EXIST = 0b100, ONLY_TABLE = 0b1000 };
    enum e_NodeType { DN_NULL = 0, DN_ENTITY = 1, DN_ARRAY = 2, DN_TABLE = 3 };

    /** @addtogroup{ Interface */
    virtual int Flush() { return 0; }
    virtual e_NodeType NodeType() const { return DN_NULL; }
    virtual size_type GetNumberOfChildren() const { return 0; }

    virtual std::shared_ptr<DataNode> Duplicate() const { return DataNode::New(); }

    virtual std::shared_ptr<DataNode> Root() { return Duplicate(); }
    virtual std::shared_ptr<DataNode> Parent() const { return Duplicate(); }
    virtual int Foreach(std::function<int(std::string, std::shared_ptr<DataNode>)> const&) { return 0; }
    virtual int Foreach(std::function<int(std::string, std::shared_ptr<const DataNode>)> const&) const { return 0; }

    virtual std::shared_ptr<DataNode> GetNode(std::string const& uri, int flag) { return Duplicate(); }
    virtual std::shared_ptr<DataNode> GetNode(std::string const& uri, int flag) const { return Duplicate(); }
    virtual std::shared_ptr<DataNode> GetNode(index_type s, int flag) { return Duplicate(); };
    virtual std::shared_ptr<DataNode> GetNode(index_type s, int flag) const { return Duplicate(); };
    virtual std::shared_ptr<DataNode> AddNode() {
        return GetNode(GetNumberOfChildren(), NEW_IF_NOT_EXIST | ADD_IF_EXIST);
    };

    virtual int DeleteNode(std::string const& s, int flag) { return 0; }
    virtual int DeleteNode(index_type s, int flag) { return DeleteNode(std::to_string(s), flag); };

    virtual std::shared_ptr<DataEntity> Get() { return DataEntity::New(); }
    virtual std::shared_ptr<DataEntity> Get() const { return DataEntity::New(); }
    virtual int Set(std::shared_ptr<DataEntity> const& v) { return 0; }
    virtual int Set(std::shared_ptr<DataNode> const& v) { return 0; }
    virtual int Add(std::shared_ptr<DataEntity> const& v) { return AddNode()->Set(v); }
    virtual int Add(std::shared_ptr<DataNode> const& v) { return AddNode()->Set(v); }

    /** @} */
    DataNode& operator[](std::string const& s) { return *GetNode(s, RECURSIVE | NEW_IF_NOT_EXIST); }
    template <typename U>
    DataNode& operator=(U const& u) {
        Set(make_data(u));
        return *this;
    }
    template <typename U>
    DataNode& operator=(std::initializer_list<U> const& u) {
        Set(make_data(u));
        return *this;
    }

    template <typename U>
    DataNode& operator=(std::initializer_list<std::initializer_list<U>> const& u) {
        Set(make_data(u));
        return *this;
    }
    template <typename U>
    DataNode& operator=(std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& u) {
        Set(make_data(u));
        return *this;
    }

    template <typename U>
    DataNode& operator+=(U const& u) {
        Add(make_data(u));
        return *this;
    }
    template <typename U>
    DataNode& operator+=(std::initializer_list<U> const& u) {
        Add(make_data(u));
        return *this;
    }
    template <typename U>
    DataNode& operator+=(std::initializer_list<std::initializer_list<U>> const& u) {
        Add(make_data(u));
        return *this;
    }
    template <typename U>
    DataNode& operator+=(std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& u) {
        Add(make_data(u));
        return *this;
    }

    template <typename URL, typename U>
    bool Check(URL const& url, U const& u) const {
        return GetNode(url, RECURSIVE)->Get()->equal(u);
    }
    bool Check(std::string const& uri) const { return Check(uri, true); }

    template <typename U>
    U as() const {
        return Get()->as<U>();
    }

    template <typename U>
    U as(U const& default_value) const {
        return Get()->as<U>(default_value);
    }
    /** @} */
};

// struct DataNode::iterator {
//    iterator(DataNode& v) : m_value_(v.shared_from_this()) {}
//    iterator(iterator const& other) : m_value_(other.m_value_) {}
//    iterator(iterator&& other) : m_value_(other.m_value_) {}
//
//    virtual ~iterator() {}
//
//    virtual void Next() {}
//    virtual bool isEqual(iterator const& other) { return false; }
//
//    DataNode operator*() { return *m_value_; }
//    iterator operator++() {
//        iterator res(*this);
//        ++res;
//        return res;
//    }
//    bool operator==(iterator const& other) { return isEqual(other); }
//    bool operator!=(iterator const& other) { return !isEqual(other); }
//
//    std::shared_ptr<DataNode> m_value_;
//};

std::pair<std::string, std::shared_ptr<DataNode>> RecursiveFindNode(std::shared_ptr<DataNode> d, std::string uri,
                                                                    int flag = 0);
// std::shared_ptr<const DataNode> RecursiveFindNode(std::shared_ptr<const DataNode> const& d, std::string const& uri,
//                                                  int flag = 0);
std::ostream& operator<<(std::ostream&, DataNode const&);
}  // namespace data
}  // namespace simpla

#endif  // SIMPLA_DATAENTRY_H
