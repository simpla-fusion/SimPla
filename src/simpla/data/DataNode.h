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
#define SP_URL_SPLIT_CHAR '/'
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

    /** @addtogroup required @{*/
    virtual eNodeType type() const = 0;
    virtual size_type size() const = 0;
    virtual std::shared_ptr<DataNode> Duplicate() const = 0;
    virtual std::shared_ptr<DataEntity> GetEntity() const = 0;

    virtual size_type Set(std::string const& uri, std::shared_ptr<DataEntity> const& v) = 0;
    virtual size_type Add(std::string const& uri, std::shared_ptr<DataEntity> const& v) = 0;
    virtual size_type Delete(std::string const& s) = 0;
    virtual std::shared_ptr<const DataNode> Get(std::string const& uri) const = 0;

    virtual size_type Foreach(
        std::function<size_type(std::string, std::shared_ptr<const DataNode>)> const& f) const = 0;

    virtual size_type Set(index_type s, std::shared_ptr<DataEntity> const& v) { return Set(std::to_string(s), v); };
    virtual size_type Add(index_type s, std::shared_ptr<DataEntity> const& v) { return Add(std::to_string(s), v); };
    virtual size_type Delete(index_type s) { return Delete(std::to_string(s)); };
    virtual std::shared_ptr<const DataNode> Get(index_type s) const { return Get(std::to_string(s)); };

    /**@ } */

    /** @addtogroup optional @{*/
    virtual int Parse(std::string const& s) { return 0; }
    virtual std::istream& Parse(std::istream& is);
    virtual std::ostream& Print(std::ostream& os, int indent) const;
    virtual int Connect(std::string const& authority, std::string const& path, std::string const& query,
                        std::string const& fragment) {
        return 0;
    };
    virtual int Disconnect() { return 0; };
    virtual bool isValid() const { return true; }
    virtual int Flush() { return 0; }

    virtual void Clear() {}

    virtual std::shared_ptr<DataNode> Root() const { return Duplicate(); }
    virtual std::shared_ptr<DataNode> Parent() const { return Duplicate(); }

    /**@ } */

    size_type Set(std::shared_ptr<DataNode> const& v);
    size_type Add(std::shared_ptr<DataNode> const& v);

    KeyValue operator[](std::string const& s) const { return Get(s); }
    KeyValue operator[](index_type s) const { return Get(s); }
    KeyValue operator[](std::string const& s) { return Get(s); }
    KeyValue operator[](index_type s) { return Get(s); }
    template <typename... Others>
    size_type SetValue(Others&&... args) {
        return Set("", DataLight::New(std::forward<Others>(args)...));
    };
    template <typename U>
    size_type SetValue(std::initializer_list<U> const& v) {
        return Set("", DataLight::New(v));
    }

    size_type SetValue(std::initializer_list<char const*> const& u) { return Set("", DataLight::New(u)); }

    template <typename U>
    size_type SetValue(std::initializer_list<std::initializer_list<U>> const& v) {
        return Set("", DataLight::New(v));
    }
    template <typename U>
    size_type SetValue(std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& v) {
        return Set("", DataLight::New(v));
    }

    template <typename... Args>
    size_type AddValue(Args&&... args) {
        return Add("", DataLight::New(std::forward<Args>(args)...));
    };

    template <typename U>
    size_type AddValue(std::initializer_list<U> const& v) {
        return Add("", DataLight::New(v));
    }
    template <typename U>
    size_type AddValue(std::initializer_list<std::initializer_list<U>> const& v) {
        return Add("", DataLight::New(v));
    }
    template <typename U>
    size_type AddValue(std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& v) {
        return Add("", DataLight::New(v));
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
        auto p = Get(url);
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

    size_type Set(KeyValue const& kv);
    size_type Add(KeyValue const& kv);
    DataNode& operator=(KeyValue const& v);
    DataNode& operator=(std::initializer_list<KeyValue> const& u);
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
            count += _CopyOut(dst[i], src == nullptr ? nullptr : src->Get(i));
        }
        return count;
    };

    template <typename U>
    size_type _CopyIn(std::shared_ptr<DataNode>& dst, U const& src, ENABLE_IF(std::rank<U>::value == 0)) const {
        return dst == nullptr ? 0 : dst->Set("", src);
    };

    template <typename U>
    size_type _CopyIn(std::shared_ptr<DataNode>& dst, U const& src, ENABLE_IF((std::rank<U>::value > 0))) const {
        size_type count = 0;
        if (dst != nullptr) {
            for (size_type i = 0; i < std::extent<U, 0>::value; ++i) {
                //                count += _CopyIn(dst == nullptr ? dst : dst->Get(i, NEW_IF_NOT_EXIST), src[i]);
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
            res = res && right != nullptr && _isEqualTo(left[i], right->Get(i));
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
std::pair<std::string, std::shared_ptr<DataNode>> RecursiveFindNode(std::shared_ptr<DataNode> d, std::string uri,
                                                                    int flag = 0);

std::ostream& operator<<(std::ostream&, DataNode const&);

struct KeyValue {
    std::shared_ptr<DataNode> m_node_;

    explicit KeyValue(std::string const& k, std::shared_ptr<DataEntity> const& v = nullptr) : m_node_(DataNode::New()) {
        m_node_->Set(k, v);
    }
    KeyValue(KeyValue const& other) : m_node_(other.m_node_) {}
    KeyValue(KeyValue&& other) : m_node_(other.m_node_) {}
    ~KeyValue() = default;

    KeyValue& operator=(KeyValue const& other) {
        m_node_ = other.m_node_;
        return *this;
    }

    template <typename U>
    KeyValue& operator=(U const& u) {
        m_node_->SetValue(0, u);
        return *this;
    }
    template <typename U>
    KeyValue& operator=(std::initializer_list<U> const& u) {
        m_node_->Clear();
        m_node_->SetValue(0, u);
        return *this;
    }
    template <typename U>
    KeyValue& operator=(std::initializer_list<std::initializer_list<U>> const& u) {
        m_node_->Clear();
        m_node_->SetValue(0, u);
        return *this;
    }
    template <typename U>
    KeyValue& operator=(std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& u) {
        m_node_->Clear();
        m_node_->SetValue(0, u);
        return *this;
    }
};
inline KeyValue operator"" _(const char* c, std::size_t n) { return KeyValue{std::string(c), make_data(true)}; }

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
                                                                                                                     \
    std::shared_ptr<_CLASS_NAME_> Self() { return std::dynamic_pointer_cast<this_type>(this->shared_from_this()); }; \
    std::shared_ptr<_CLASS_NAME_> Self() const {                                                                     \
        return std::dynamic_pointer_cast<this_type>(const_cast<this_type*>(this)->shared_from_this());               \
    };                                                                                                               \
    std::shared_ptr<DataNode> Root() const override;                                                                 \
    std::shared_ptr<DataNode> Parent() const override;                                                               \
                                                                                                                     \
    eNodeType type() const override;                                                                                 \
    size_type size() const override;                                                                                 \
                                                                                                                     \
    std::shared_ptr<DataNode> Duplicate() const override;                                                            \
                                                                                                                     \
    std::shared_ptr<DataEntity> GetEntity() const override;                                                          \
                                                                                                                     \
    size_type Set(std::string const& uri, std::shared_ptr<DataEntity> const& v) override;                            \
    size_type Add(std::string const& uri, std::shared_ptr<DataEntity> const& v) override;                            \
    size_type Delete(std::string const& s) override;                                                                 \
    std::shared_ptr<const DataNode> Get(std::string const& uri) const override;                                      \
                                                                                                                     \
    size_type Foreach(std::function<size_type(std::string, std::shared_ptr<const DataNode>)> const& f) const override;

void Clear() {}

}  // namespace data
}  // namespace simpla

#endif  // SIMPLA_DATAENTRY_H
