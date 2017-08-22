//
// Created by salmon on 17-8-18.
//

#ifndef SIMPLA_DATAENTRY_H
#define SIMPLA_DATAENTRY_H

#include "simpla/SIMPLA_config.h"

#include <memory>
#include "DataLight.h"
#include "DataUtilities.h"
#include "simpla/utilities/Factory.h"
#include "simpla/utilities/Log.h"
#include "simpla/utilities/ObjectHead.h"
namespace simpla {
namespace data {

class DataEntity;
class KeyValue;

class DataNode : public Factory<DataNode>, public std::enable_shared_from_this<DataNode> {
    SP_OBJECT_BASE(DataNode);
    static int s_num_of_pre_registered_;

   protected:
    DataNode();

   public:
    virtual ~DataNode();
    DataNode(DataNode const& other) = delete;
    DataNode(DataNode&& other) = delete;

    static std::shared_ptr<DataNode> New(std::string const& uri = "");

    virtual int Connect(std::string const& authority, std::string const& path, std::string const& query,
                        std::string const& fragment) = 0;
    virtual int Disconnect() = 0;
    virtual bool isValid() const { return true; }

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
    virtual int Foreach(std::function<int(std::string, std::shared_ptr<DataNode>)> const&) const { return 0; }

    virtual std::shared_ptr<DataNode> GetNode(std::string const& uri, int flag) { return Duplicate(); }
    virtual std::shared_ptr<DataNode> GetNode(std::string const& uri, int flag) const { return Duplicate(); }
    virtual std::shared_ptr<DataNode> GetNode(index_type s, int flag) { return Duplicate(); };
    virtual std::shared_ptr<DataNode> GetNode(index_type s, int flag) const { return Duplicate(); };
    virtual std::shared_ptr<DataNode> AddNode() {
        return GetNode(GetNumberOfChildren(), NEW_IF_NOT_EXIST | ADD_IF_EXIST);
    };

    virtual int DeleteNode(std::string const& s, int flag) { return 0; }
    virtual int DeleteNode(index_type s, int flag) { return DeleteNode(std::to_string(s), flag); };
    virtual void Clear() {}

    virtual std::shared_ptr<DataEntity> Get();
    virtual std::shared_ptr<DataEntity> Get() const;
    virtual int Set(std::shared_ptr<DataEntity> const& v);
    virtual int Add(std::shared_ptr<DataEntity> const& v);
    virtual int Set(std::shared_ptr<DataNode> const& v);
    virtual int Add(std::shared_ptr<DataNode> const& v);

    /** @} */
    DataNode& operator[](std::string const& s) { return *GetNode(s, RECURSIVE | NEW_IF_NOT_EXIST); }

    template <typename U, typename... Args>
    int SetValue(Args&&... args) {
        return Set(DataLightT<U>::New(std::forward<Args>(args)...));
    };
    template <typename U, typename V>
    int SetValue(std::initializer_list<V> const& v) {
        return Set(DataLightT<U>::New(v));
    }

    template <typename U>
    int SetValue(std::initializer_list<char const*> const& u) {
        int count = 0;
        for (auto const& v : u) { count += SetValue<std::string>(v); }
        return count;
    }

    template <typename U, typename V>
    int SetValue(std::initializer_list<std::initializer_list<V>> const& v) {
        return Set(DataLightT<U>::New(v));
    }
    template <typename U, typename V>
    int SetValue(std::initializer_list<std::initializer_list<std::initializer_list<V>>> const& v) {
        return Set(DataLightT<U>::New(v));
    }

    template <typename U, typename... Args>
    int AddValue(Args&&... args) {
        return Add(DataLightT<U>::New(std::forward<Args>(args)...));
    };

    template <typename U>
    int AddValue(std::initializer_list<char const*> const& u) {
        int count = 0;
        for (auto const& v : u) { count += AddValue<std::string>(v); }
        return count;
    }

    template <typename U, typename V>
    int AddValue(std::initializer_list<V> const& v) {
        return Add(DataLightT<U>::New(v));
    }
    template <typename U, typename V>
    int AddValue(std::initializer_list<std::initializer_list<V>> const& v) {
        return Add(DataLightT<U>::New(v));
    }
    template <typename U, typename V>
    int AddValue(std::initializer_list<std::initializer_list<std::initializer_list<V>>> const& v) {
        return Add(DataLightT<U>::New(v));
    }
    template <typename U>
    U GetValue() const {
        return Get()->as<U>();
    };
    template <typename U>
    U GetValue(U const& default_value) const {
        return Get()->as<U>(default_value);
    };
    template <typename URL, typename U>
    bool Check(URL const& url, U const& u) const {
        return GetNode(url, RECURSIVE)->Get()->equal(u);
    }
    bool Check(std::string const& uri) const { return Check(uri, true); }

    template <typename U>
    U as() const {
        return GetValue<U>();
    }

    template <typename U>
    U as(U const& default_value) const {
        return GetValue<U>(default_value);
    }

    template <typename U>
    DataNode& operator=(U const& u) {
        SetValue<U>(u);
        return *this;
    }
    DataNode& operator=(char const* u) {
        SetValue<std::string>(u);
        return *this;
    }

    DataNode& operator=(std::initializer_list<char const*> const& u) {
        SetValue<std::string>(u);
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
    DataNode& operator+=(char const* u) {
        AddValue<std::string>(u);
        return *this;
    }

    template <typename U>
    DataNode& operator+=(std::initializer_list<U> const& u) {
        Add(make_data(u));
        return *this;
    }
    DataNode& operator+=(std::initializer_list<char const*> const& u) {
        AddValue<std::string>(u);
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

    int Set(KeyValue const& kv);
    int Add(KeyValue const& kv);
    int SetValue(KeyValue const& kv);
    int AddValue(KeyValue const& kv);
    template <typename U>
    int SetValue(std::initializer_list<KeyValue> const& u);
    template <typename U>
    int AddValue(std::initializer_list<KeyValue> const& u);
    DataNode& operator=(KeyValue const& v);
    DataNode& operator=(std::initializer_list<KeyValue> const& u);
    DataNode& operator+=(KeyValue const& u);
    DataNode& operator+=(std::initializer_list<KeyValue> const& u);
    /** @} */
};

class KeyValue {
    std::string m_key_;
    std::shared_ptr<DataNode> m_node_;

   public:
    explicit KeyValue(std::string const& k, std::shared_ptr<DataEntity> const& v = nullptr)
        : m_key_(k), m_node_(DataNode::New()) {
        m_node_->GetNode(m_key_, DataNode::RECURSIVE | DataNode::NEW_IF_NOT_EXIST)->Set(v);
    }
    KeyValue(KeyValue const& other) : m_key_(other.m_key_), m_node_(other.m_node_) {}
    KeyValue(KeyValue&& other) : m_key_(other.m_key_), m_node_(other.m_node_) {}
    ~KeyValue() = default;
    std::shared_ptr<DataNode>& GetNode() { return m_node_; }
    std::shared_ptr<DataNode> const& GetNode() const { return m_node_; }

    KeyValue& operator=(KeyValue const& other) {
        m_node_ = other.m_node_;
        return *this;
    }

    template <typename U>
    KeyValue& operator=(U const& u) {
        (*m_node_)[m_key_] = u;
        return *this;
    }
    template <typename U>
    KeyValue& operator=(std::initializer_list<U> const& u) {
        m_node_->Clear();
        (*m_node_)[m_key_] = u;
        return *this;
    }
    template <typename U>
    KeyValue& operator=(std::initializer_list<std::initializer_list<U>> const& u) {
        (*m_node_)[m_key_] = u;
        return *this;
    }
    template <typename U>
    KeyValue& operator=(std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& u) {
        (*m_node_)[m_key_] = u;
        return *this;
    }
};
inline KeyValue operator"" _(const char* c, std::size_t n) { return KeyValue{std::string(c), make_data(true)}; }

inline int DataNode::Set(KeyValue const& kv) { return Set(kv.GetNode()); }
inline int DataNode::Add(KeyValue const& kv) { return Add(kv.GetNode()); }
inline int DataNode::SetValue(KeyValue const& kv) { return Set(kv.GetNode()); }
inline int DataNode::AddValue(KeyValue const& kv) { return Add(kv.GetNode()); }
template <typename U>
int DataNode::SetValue(std::initializer_list<KeyValue> const& u) {
    int count = 0;
    for (auto const& v : u) { count += Set(v); }
    return count;
}
template <typename U>
int DataNode::AddValue(std::initializer_list<KeyValue> const& u) {
    int count = 0;
    for (auto const& v : u) { count += Set(v); }
    return count;
}
inline DataNode& DataNode::operator=(KeyValue const& v) {
    Set(v);
    return *this;
}
inline DataNode& DataNode::operator=(std::initializer_list<KeyValue> const& u) {
    for (auto const& v : u) { Set(v); }
    return *this;
}
inline DataNode& DataNode::operator+=(KeyValue const& u) {
    Set(u);
    return *this;
}
inline DataNode& DataNode::operator+=(std::initializer_list<KeyValue> const& u) {
    for (auto const& v : u) { Set(v); }
    return *this;
}
std::pair<std::string, std::shared_ptr<DataNode>> RecursiveFindNode(std::shared_ptr<DataNode> d, std::string uri,
                                                                    int flag = 0);
// std::shared_ptr<const DataNode> RecursiveFindNode(std::shared_ptr<const DataNode> const& d, std::string const& uri,
//                                                  int flag = 0);
std::ostream& operator<<(std::ostream&, DataNode const&);

#define SP_DATA_NODE_HEAD(_CLASS_NAME_)                                                                \
    SP_DEFINE_FANCY_TYPE_NAME(_CLASS_NAME_, DataNode)                                                  \
    struct pimpl_s;                                                                                    \
    pimpl_s* m_pimpl_ = nullptr;                                                                       \
                                                                                                       \
   protected:                                                                                          \
    _CLASS_NAME_();                                                                                    \
                                                                                                       \
    explicit _CLASS_NAME_(_CLASS_NAME_ const& other) = delete;                                         \
    explicit _CLASS_NAME_(_CLASS_NAME_&& other) = delete;                                              \
    _CLASS_NAME_& operator=(_CLASS_NAME_ const& other) = delete;                                       \
    _CLASS_NAME_& operator=(_CLASS_NAME_&& other) = delete;                                            \
                                                                                                       \
   public:                                                                                             \
    ~_CLASS_NAME_() override;                                                                          \
                                                                                                       \
    int Connect(std::string const& authority, std::string const& path, std::string const& query,       \
                std::string const& fragment) override;                                                 \
    int Disconnect() override;                                                                         \
    bool isValid() const override;                                                                     \
    int Flush() override;                                                                              \
    template <typename... Args>                                                                        \
    static std::shared_ptr<this_type> New(Args&&... args) {                                            \
        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...));                 \
    }                                                                                                  \
    std::shared_ptr<DataNode> Duplicate() const override;                                              \
    size_type GetNumberOfChildren() const override;                                                    \
                                                                                                       \
    e_NodeType NodeType() const override;                                                              \
                                                                                                       \
    std::shared_ptr<DataNode> Root() override;                                                         \
    std::shared_ptr<DataNode> Parent() const override;                                                 \
                                                                                                       \
    int Foreach(std::function<int(std::string, std::shared_ptr<DataNode>)> const& fun) override;       \
    int Foreach(std::function<int(std::string, std::shared_ptr<DataNode>)> const& fun) const override; \
                                                                                                       \
    std::shared_ptr<DataNode> GetNode(std::string const& uri, int flag) override;                      \
    std::shared_ptr<DataNode> GetNode(std::string const& uri, int flag) const override;                \
    std::shared_ptr<DataNode> GetNode(index_type s, int flag) override;                                \
    std::shared_ptr<DataNode> GetNode(index_type s, int flag) const override;                          \
    int DeleteNode(std::string const& uri, int flag) override;                                         \
    void Clear() override;                                                                             \
                                                                                                       \
    std::shared_ptr<DataEntity> Get() override;                                                        \
    std::shared_ptr<DataEntity> Get() const override;                                                  \
    int Set(std::shared_ptr<DataEntity> const& v) override;                                            \
    int Add(std::shared_ptr<DataEntity> const& v) override;
}  // namespace data
}  // namespace simpla

#endif  // SIMPLA_DATAENTRY_H
