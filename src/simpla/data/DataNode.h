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

    enum { RECURSIVE = 0b01, NEW_IF_NOT_EXIST = 0b010, ADD_IF_EXIST = 0b100, ONLY_TABLE = 0b1000 };
    enum eNodeType { DN_NULL = 0, DN_ENTITY = 1, DN_TABLE = 2, DN_ARRAY = 3, DN_FUNCTION = 4 };
    enum eDataIOStatus {};
    static std::shared_ptr<DataNode> New(std::string const& uri = "");

    /** @addtogroup optional @{*/
    virtual int Parse(std::string const& s) { return 0; }
    virtual std::istream& Parse(std::istream& is);
    virtual std::ostream& Print(std::ostream& os, int indent) const;
    virtual int Connect(std::string const& authority, std::string const& path, std::string const& query,
                        std::string const& fragment) = 0;
    virtual int Disconnect() = 0;
    virtual bool isValid() const { return true; }
    virtual int Flush() { return 0; }
    /**@ } */

    /** @addtogroup required @{*/

    /** @addtogroup{ Interface */
    virtual eNodeType type() const { return DN_NULL; }
    virtual size_type size() const { return 0; }

    virtual std::shared_ptr<DataNode> Duplicate() const { return DataNode::New(); }

    virtual std::shared_ptr<DataNode> Root() const { return Duplicate(); }
    virtual std::shared_ptr<DataNode> Parent() const { return Duplicate(); }
    virtual size_type Foreach(std::function<size_type(std::string, std::shared_ptr<DataNode>)> const& f) { return 0; }
    virtual size_type Foreach(std::function<size_type(std::string, std::shared_ptr<DataNode>)> const& f) const {
        return 0;
    }

    virtual std::shared_ptr<DataNode> GetNode(std::string const& uri, int flag) { return Duplicate(); }
    virtual std::shared_ptr<DataNode> GetNode(std::string const& uri, int flag) const { return Duplicate(); }
    virtual std::shared_ptr<DataNode> GetNode(index_type s, int flag) { return Duplicate(); };
    virtual std::shared_ptr<DataNode> GetNode(index_type s, int flag) const { return Duplicate(); };

    virtual size_type SetNode(std::shared_ptr<DataNode> const& v);
    virtual size_type AddNode(std::shared_ptr<DataNode> const& v);

    virtual std::shared_ptr<DataNode> AddNode() { return GetNode(size(), NEW_IF_NOT_EXIST | ADD_IF_EXIST); };
    virtual size_type DeleteNode(std::string const& s, int flag) { return 0; }
    virtual size_type DeleteNode(index_type s, int flag) { return DeleteNode(std::to_string(s), flag); };
    virtual void Clear() {}

    virtual std::shared_ptr<DataEntity> GetEntity() const;
    virtual size_type SetEntity(std::shared_ptr<DataEntity> const& v);
    virtual size_type AddEntity(std::shared_ptr<DataEntity> const& v);

    /**@ } */

    /** @} */
    std::shared_ptr<DataNode> operator[](std::string const& s) { return GetNode(s); }
    std::shared_ptr<DataNode> operator[](std::string const& s) const { return GetNode(s); }

    std::shared_ptr<DataNode> GetNode(std::string const& uri) { return GetNode(uri, RECURSIVE | NEW_IF_NOT_EXIST); }
    std::shared_ptr<DataNode> GetNode(std::string const& uri) const { return GetNode(uri, RECURSIVE); }
    std::shared_ptr<DataNode> GetNode(size_type s) { return GetNode(s, NEW_IF_NOT_EXIST); }
    std::shared_ptr<DataNode> GetNode(size_type s) const { return GetNode(s, 0); }

    template <typename... Others>
    size_type SetValue(Others&&... args) {
        return SetEntity(DataLight::New(std::forward<Others>(args)...));
    };
    template <typename U>
    size_type SetValue(std::initializer_list<U> const& v) {
        return SetEntity(DataLight::New(v));
    }

    size_type SetValue(std::initializer_list<char const*> const& u) { return SetEntity(DataLight::New(u)); }

    template <typename U>
    size_type SetValue(std::initializer_list<std::initializer_list<U>> const& v) {
        return SetEntity(DataLight::New(v));
    }
    template <typename U>
    size_type SetValue(std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& v) {
        return SetEntity(DataLight::New(v));
    }

    template <typename... Args>
    size_type AddValue(Args&&... args) {
        return AddEntity(DataLight::New(std::forward<Args>(args)...));
    };

    template <typename U>
    size_type AddValue(std::initializer_list<U> const& v) {
        return AddEntity(DataLight::New(v));
    }
    template <typename U>
    size_type AddValue(std::initializer_list<std::initializer_list<U>> const& v) {
        return AddEntity(DataLight::New(v));
    }
    template <typename U>
    size_type AddValue(std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& v) {
        return AddEntity(DataLight::New(v));
    }
    template <typename U>
    U GetValue(ENABLE_IF((traits::is_light_data<U>::value))) const {
        U res;
        if (CopyOut(res) == 0) {
            //            FIXME << "BAD_CAST";
            res = std::numeric_limits<U>::signaling_NaN();
        }
        return res;
    };

    std::shared_ptr<DataBlock> GetData() const { return std::dynamic_pointer_cast<DataBlock>(GetEntity()); }

    template <typename U>
    std::shared_ptr<DataBlockT<traits::value_type_t<U>>> GetValue(ENABLE_IF((!traits::is_light_data<U>::value))) const {
        return std::dynamic_pointer_cast<DataBlockT<traits::value_type_t<U>>>(GetEntity());
    }

    template <typename U>
    U GetValue(U const& default_value) const {
        U res;
        return CopyOut(res) > 0 ? res : default_value;
    };
    template <typename URL, typename U>
    bool Check(URL const& url, U const& u) const {
        auto p = GetNode(url, RECURSIVE);
        return p != nullptr && p->isEqualTo(u);
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
        SetValue(u);
        return *this;
    }
    DataNode& operator=(char const* u) {
        SetValue(std::string(u));
        return *this;
    }

    template <typename U>
    DataNode& operator=(std::initializer_list<U> const& u) {
        SetValue(u);
        return *this;
    }

    template <typename U>
    DataNode& operator=(std::initializer_list<std::initializer_list<U>> const& u) {
        SetValue(u);
        return *this;
    }
    template <typename U>
    DataNode& operator=(std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& u) {
        SetValue(u);
        return *this;
    }

    template <typename U>
    DataNode& operator+=(U const& u) {
        AddValue(u);
        return *this;
    }

    template <typename U>
    DataNode& operator+=(std::initializer_list<U> const& u) {
        AddValue(u);
        return *this;
    }
    DataNode& operator+=(std::initializer_list<char const*> const& u) {
        AddValue(u);
        return *this;
    }

    template <typename U>
    DataNode& operator+=(std::initializer_list<std::initializer_list<U>> const& u) {
        AddValue(u);
        return *this;
    }
    template <typename U>
    DataNode& operator+=(std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& u) {
        AddValue(u);
        return *this;
    }

    size_type SetNode(KeyValue const& kv);
    size_type AddNode(KeyValue const& kv);
    size_type SetNode(std::initializer_list<KeyValue> const& u);
    size_type AddNode(std::initializer_list<KeyValue> const& u);
    DataNode& operator=(KeyValue const& v);
    DataNode& operator=(std::initializer_list<KeyValue> const& u);
    DataNode& operator+=(KeyValue const& u);
    DataNode& operator+=(std::initializer_list<KeyValue> const& u);
    /** @} */

    //******************************************************************************************************
   private:
    template <typename U>
    size_type _CopyOut(U& dst, std::shared_ptr<const DataNode> const& src, ENABLE_IF(std::rank<U>::value == 0)) const {
        size_type count = 0;
        if (auto p = (src == nullptr) ? nullptr : std::dynamic_pointer_cast<DataLightT<U>>(src->GetEntity())) {
            dst = p->value();
            ++count;
        } else {
            dst = std::numeric_limits<U>::signaling_NaN();
        }
        return count;
    };

    template <typename U>
    size_type _CopyOut(U& dst, std::shared_ptr<const DataNode> const& src, ENABLE_IF((std::rank<U>::value > 0))) const {
        size_type count = 0;
        for (size_type i = 0; i < std::extent<U, 0>::value; ++i) {
            count += _CopyOut(dst[i], src == nullptr ? nullptr : src->GetNode(i));
        }
        return count;
    };

    template <typename U>
    size_type _CopyIn(std::shared_ptr<DataNode>& dst, U const& src, ENABLE_IF(std::rank<U>::value == 0)) const {
        return dst == nullptr ? 0 : dst->SetEntity(src);
    };

    template <typename U>
    size_type _CopyIn(std::shared_ptr<DataNode>& dst, U const& src, ENABLE_IF((std::rank<U>::value > 0))) const {
        size_type count = 0;
        if (dst != nullptr) {
            for (size_type i = 0; i < std::extent<U, 0>::value; ++i) {
                count += _CopyIn(dst == nullptr ? dst : dst->GetNode(i, NEW_IF_NOT_EXIST), src[i]);
            }
        }
        return count;
    };
    template <typename U>
    static bool _isEqualTo(U const& left, std::shared_ptr<const DataNode> const& right,
                           ENABLE_IF(std::rank<U>::value == 0)) {
        return right != nullptr && right->GetEntity() != nullptr && right->as<U>() == left;
    };

    template <typename U>
    static bool _isEqualTo(U const& left, std::shared_ptr<const DataNode> const& right,
                           ENABLE_IF((std::rank<U>::value > 0))) {
        bool res = true;
        for (size_type i = 0; res && i < std::extent<U, 0>::value; ++i) {
            res = res && right != nullptr && _isEqualTo(left[i], right->GetNode(i));
        }
        return res;
    };

   public:
    template <typename U>
    size_type CopyOut(U& res) const {
        size_type count = 0;
        if (auto p = std::dynamic_pointer_cast<DataLightT<U>>(GetEntity())) {
            res = p->value();
            count = 1;
        } else if (auto p = std::dynamic_pointer_cast<DataLightT<traits::value_type_t<U>*>>(GetEntity())) {
            count = p->CopyOut(res);
        } else {
            count = _CopyOut(res, shared_from_this());
        }
        return count;
    };
    template <typename U>
    size_type CopyIn(U const& dst) {
        return _CopyIn(dst, shared_from_this());
    };
    template <typename U>
    bool isEqualTo(U const& dst) const {
        bool res = false;
        auto p0 = GetEntity();
        if (p0 == nullptr) {
        } else if (auto p = std::dynamic_pointer_cast<DataLightT<U>>(p0)) {
            res = dst == p->value();
        } else if (auto p = std::dynamic_pointer_cast<DataLightT<traits::value_type_t<U>*>>(p0)) {
            res = p->isEqualTo(dst);
        } else {
            res = _isEqualTo(dst, shared_from_this());
        }
        return res;
    };
    template <typename U>
    bool operator==(U const& other) const {
        return isEqualTo(other);
    }
    //******************************************************************************************************
};

class KeyValue {
    std::string m_key_;
    std::shared_ptr<DataNode> m_node_;

   public:
    explicit KeyValue(std::string const& k, std::shared_ptr<DataEntity> const& v = nullptr)
        : m_key_(k), m_node_(DataNode::New()) {
        m_node_->GetNode(m_key_, DataNode::RECURSIVE | DataNode::NEW_IF_NOT_EXIST)->SetEntity(v);
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
        *m_node_->GetNode(m_key_) = (u);
        return *this;
    }
    template <typename U>
    KeyValue& operator=(std::initializer_list<U> const& u) {
        m_node_->Clear();
        *m_node_->GetNode(m_key_) = (u);
        return *this;
    }
    template <typename U>
    KeyValue& operator=(std::initializer_list<std::initializer_list<U>> const& u) {
        m_node_->Clear();
        *m_node_->GetNode(m_key_) = (u);
        return *this;
    }
    template <typename U>
    KeyValue& operator=(std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& u) {
        m_node_->Clear();
        *m_node_->GetNode(m_key_) = (u);
        return *this;
    }
};
inline KeyValue operator"" _(const char* c, std::size_t n) { return KeyValue{std::string(c), make_data(true)}; }

inline size_type DataNode::SetNode(KeyValue const& kv) { return SetNode(kv.GetNode()); }
inline size_type DataNode::AddNode(KeyValue const& kv) { return AddNode(kv.GetNode()); }
inline size_type DataNode::SetNode(std::initializer_list<KeyValue> const& u) {
    size_type count = 0;
    for (auto const& v : u) { count += SetNode(v); }
    return count;
}
inline size_type DataNode::AddNode(std::initializer_list<KeyValue> const& u) {
    size_type count = 0;
    for (auto const& v : u) { count += AddNode(v); }
    return count;
}
inline DataNode& DataNode::operator=(KeyValue const& v) {
    SetNode(v);
    return *this;
}
inline DataNode& DataNode::operator=(std::initializer_list<KeyValue> const& u) {
    SetNode(u);
    return *this;
}
inline DataNode& DataNode::operator+=(KeyValue const& u) {
    AddNode(u);
    return *this;
}
inline DataNode& DataNode::operator+=(std::initializer_list<KeyValue> const& u) {
    AddNode(u);
    return *this;
}
std::pair<std::string, std::shared_ptr<DataNode>> RecursiveFindNode(std::shared_ptr<DataNode> d, std::string uri,
                                                                    int flag = 0);
// std::shared_ptr<const DataNode> RecursiveFindNode(std::shared_ptr<const DataNode> const& d, std::string const& uri,
//                                                  int flag = 0);
std::ostream& operator<<(std::ostream&, DataNode const&);

#define SP_DATA_NODE_HEAD(_CLASS_NAME_)                                                                              \
    SP_DEFINE_FANCY_TYPE_NAME(_CLASS_NAME_, DataNode)                                                                \
    struct pimpl_s;                                                                                                  \
    pimpl_s* m_pimpl_ = nullptr;                                                                                     \
                                                                                                                     \
    explicit _CLASS_NAME_(pimpl_s*);                                                                                 \
                                                                                                                     \
   protected:                                                                                                        \
    _CLASS_NAME_();                                                                                                  \
                                                                                                                     \
   public:                                                                                                           \
    explicit _CLASS_NAME_(_CLASS_NAME_ const& other) = delete;                                                       \
    explicit _CLASS_NAME_(_CLASS_NAME_&& other) = delete;                                                            \
    _CLASS_NAME_& operator=(_CLASS_NAME_ const& other) = delete;                                                     \
    _CLASS_NAME_& operator=(_CLASS_NAME_&& other) = delete;                                                          \
                                                                                                                     \
    ~_CLASS_NAME_() override;                                                                                        \
                                                                                                                     \
    template <typename... Args>                                                                                      \
    static std::shared_ptr<this_type> New(Args&&... args) {                                                          \
        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...));                               \
    }                                                                                                                \
    std::shared_ptr<DataNode> Duplicate() const override;                                                            \
    size_type size() const override;                                                                                 \
    eNodeType type() const override;                                                                                 \
    int Connect(std::string const& authority, std::string const& path, std::string const& query,                     \
                std::string const& fragment) override;                                                               \
    int Disconnect() override;                                                                                       \
    bool isValid() const override;                                                                                   \
    int Flush() override;                                                                                            \
                                                                                                                     \
    std::shared_ptr<_CLASS_NAME_> Self() { return std::dynamic_pointer_cast<this_type>(this->shared_from_this()); }; \
    std::shared_ptr<_CLASS_NAME_> Self() const {                                                                     \
        return std::dynamic_pointer_cast<this_type>(const_cast<this_type*>(this)->shared_from_this());               \
    };                                                                                                               \
    std::shared_ptr<DataNode> Root() const override;                                                                 \
    std::shared_ptr<DataNode> Parent() const override;                                                               \
                                                                                                                     \
    size_type Foreach(std::function<size_type(std::string, std::shared_ptr<DataNode>)> const& fun) override;         \
    size_type Foreach(std::function<size_type(std::string, std::shared_ptr<DataNode>)> const& fun) const override;   \
                                                                                                                     \
    std::shared_ptr<DataNode> GetNode(std::string const& uri, int flag) override;                                    \
    std::shared_ptr<DataNode> GetNode(std::string const& uri, int flag) const override;                              \
    std::shared_ptr<DataNode> GetNode(index_type s, int flag) override;                                              \
    std::shared_ptr<DataNode> GetNode(index_type s, int flag) const override;                                        \
    size_type DeleteNode(std::string const& uri, int flag) override;                                                 \
    void Clear() override;                                                                                           \
    std::shared_ptr<DataEntity> GetEntity() const override;                                                          \
    size_type SetEntity(std::shared_ptr<DataEntity> const& v) override;                                              \
    size_type AddEntity(std::shared_ptr<DataEntity> const& v) override;

}  // namespace data
}  // namespace simpla

#endif  // SIMPLA_DATAENTRY_H
