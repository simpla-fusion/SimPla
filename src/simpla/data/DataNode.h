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
class DataNode;
template <typename U>
size_type CopyFromData(U& dst, std::shared_ptr<DataNode> const& src);
template <typename U>
size_type CopyToData(std::shared_ptr<DataNode> dst, U const& src);
template <typename U>
bool EqualTo(U const& dst, std::shared_ptr<DataNode> const& src);

class DataNode : public Factory<DataNode>, public std::enable_shared_from_this<DataNode> {
   public:
    static std::string GetFancyTypeName_s() { return "DataNode"; }
    virtual std::string GetFancyTypeName() const { return GetFancyTypeName_s(); }
    static int s_num_of_pre_registered_;

   private:
    typedef Factory<DataNode> base_type;
    typedef DataNode this_type;

   public:
   public:
    enum eNodeType { DN_NULL = 0, DN_ENTITY = 1, DN_TABLE = 2, DN_ARRAY = 3, DN_FUNCTION = 4 };
    eNodeType m_type_;

   protected:
    std::shared_ptr<DataEntity> m_entity_ = nullptr;

    explicit DataNode(eNodeType etype = DN_TABLE);
    explicit DataNode(std::shared_ptr<DataEntity> v) : m_type_(DataNode::DN_ENTITY), m_entity_(std::move(v)) {}

   public:
    ~DataNode() override;
    static std::shared_ptr<DataNode> New(std::string const& uri);
    static std::shared_ptr<DataNode> New(eNodeType e_type, std::string const& uri = "");
    static std::shared_ptr<DataNode> New(std::shared_ptr<DataEntity> v) {
        return std::shared_ptr<DataNode>(new DataNode(v));
    }

    eNodeType type() const;

    std::shared_ptr<DataNode> GetRoot() const {
        return GetParent() == nullptr ? const_cast<DataNode*>(this)->shared_from_this() : GetParent()->GetRoot();
    }
    bool isRoot() const { return GetParent() == nullptr; }
    void SetParent(std::shared_ptr<DataNode>) {}
    std::shared_ptr<DataNode> GetParent() const { return nullptr; }
    /** @addtogroup required @{*/
    virtual std::shared_ptr<DataNode> Duplicate() const;
    virtual std::shared_ptr<DataNode> CreateNode(eNodeType e_type) const;
    virtual std::shared_ptr<DataNode> CreateNode(std::string const& url, eNodeType e_type);

    std::shared_ptr<DataEntity> GetEntity(int N) const {
        std::shared_ptr<DataEntity> res = GetEntity();
        if (res == nullptr && size() > 0) { res = Get(N)->GetEntity(); }
        return res;
    }

    virtual std::shared_ptr<DataEntity> GetEntity() const { return m_entity_; }
    virtual void SetEntity(std::shared_ptr<DataEntity> e) { m_entity_ = e; }

    virtual size_type size() const;
    virtual bool empty() const { return size() == 0; }

    virtual size_type Set(std::string const& uri, const std::shared_ptr<DataNode>& v);
    virtual size_type Add(std::string const& uri, const std::shared_ptr<DataNode>& v);
    virtual size_type Delete(std::string const& s);
    virtual std::shared_ptr<DataNode> Get(std::string const& uri) const;
    virtual void Foreach(std::function<void(std::string const&, std::shared_ptr<DataNode> const&)> const& f) const;

    virtual size_type Set(index_type s, const std::shared_ptr<DataNode>& v);
    virtual size_type Add(index_type s, const std::shared_ptr<DataNode>& v);
    virtual size_type Delete(index_type s);
    virtual std::shared_ptr<DataNode> Get(index_type s) const;

    virtual size_type Add(const std::shared_ptr<DataNode>& v);
    virtual size_type Set(const std::shared_ptr<DataNode>& v);
    //    virtual size_type SetValue(const std::shared_ptr<DataNode>& v) { return Set(v); }
    //    virtual size_type AddValue(const std::shared_ptr<DataNode>& v) { return Add(v); }
    //    size_type Set(std::string const& uri, const std::shared_ptr<DataEntity>& v) {
    //        auto p = CreateNode(DN_ENTITY);
    //        p->SetEntity(v);
    //        return Set(uri, p);
    //    }
    //    size_type Set(index_type s, const std::shared_ptr<DataEntity>& v) {
    //        auto p = CreateNode(DN_ENTITY);
    //        p->SetEntity(v);
    //        return Set(s, p);
    //    }
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

    /**@ } */
    size_type SetValue(std::string const& s) { return 0; };
    template <typename... Args>
    size_type SetValue(std::string const& s, Args&&... args) {
        return Set(s, DataNode::New(DataLight::New(std::forward<Args>(args)...)));
    };
    template <typename U>
    size_type SetValue(std::string const& s, U const& v) {
        return Set(s, DataNode::New(DataLight::New(v)));
    }
    template <typename U>
    size_type SetValue(std::string const& s, std::initializer_list<U> const& v) {
        return Set(s, DataNode::New(DataLight::New(v)));
    }
    template <typename U>
    size_type SetValue(std::string const& s, std::initializer_list<std::initializer_list<U>> const& v) {
        return Set(s, DataNode::New(DataLight::New(v)));
    }
    template <typename U>
    size_type SetValue(std::string const& s,
                       std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& v) {
        return Set(s, DataNode::New(DataLight::New(v)));
    }

    template <typename U>
    size_type AddValue(std::string const& s, U const& u) {
        return Add(s, DataNode::New(DataLight::New(u)));
    };

    template <typename U>
    size_type AddValue(std::string const& s, std::initializer_list<U> const& v) {
        return Add(s, DataNode::New(DataLight::New(v)));
    }
    template <typename U>
    size_type AddValue(std::string const& s, std::initializer_list<std::initializer_list<U>> const& v) {
        return Add(s, DataNode::New(DataLight::New(v)));
    }
    template <typename U>
    size_type AddValue(std::string const& s,
                       std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& v) {
        return Add(s, DataNode::New(DataLight::New(v)));
    }

    //    template <typename U>
    //    size_type SetValue(std::string const& s, U const& u, ENABLE_IF(!traits::is_light_data<U>::value)) {
    //        return Set(s, DataNode::New(DataBlock::New(u)));
    //    };
    //    template <typename U>
    //    size_type AddValue(std::string const& s, U const& u, ENABLE_IF(!traits::is_light_data<U>::value)) {
    //        DOMAIN_ERROR;
    //        return 0;
    //    };
    template <typename U>
    U GetValue(std::string const& url) const {
        U res;
        if (CopyFromData(res, Get(url)) == 0) { res = std::numeric_limits<U>::signaling_NaN(); }
        return res;
    };

    template <typename U>
    U GetValue(std::string const& url, U default_value) const {
        if (auto p = Get(url)) { CopyFromData(default_value, p); }
        return (default_value);
    };
    std::string GetValue(std::string const& url, char const* default_value) const {
        std::string res(default_value);
        if (auto p = Get(url)) { CopyFromData(res, p); }
        return (res);
    };

    template <typename U>
    bool Check(std::string const& url, U const& u) const {
        return EqualTo(u, Get(url));
    }

    bool Check(std::string const& url, char const* u) const {
        return EqualTo(std::string(u), Get(url));
        ;
    }
    bool Check(std::string const& uri) const { return Check(uri, true); }

    /** @} */

    //******************************************************************************************************
    size_type SetValue() { return 0; }

    size_type SetValue(KeyValue const& v);
    template <typename... Others>
    size_type SetValue(KeyValue const& v, Others&&... others) {
        return SetValue(v) + SetValue(std::forward<Others>(others)...);
    };
    size_type SetValue(std::initializer_list<KeyValue> const& v);

    size_type SetValue(std::string const& s, KeyValue const& v);
    size_type SetValue(std::string const& s, std::initializer_list<KeyValue> const& v);
    size_type SetValue(std::string const& s, std::initializer_list<std::initializer_list<KeyValue>> const& v);
    size_type AddValue(std::string const& s, KeyValue const& v);
    size_type AddValue(std::string const& s, std::initializer_list<KeyValue> const& v);
    size_type AddValue(std::string const& s, std::initializer_list<std::initializer_list<KeyValue>> const& v);
};

std::ostream& operator<<(std::ostream&, DataNode const&);

struct KeyValue {
    std::string m_key_;
    std::shared_ptr<DataNode> m_node_;

    explicit KeyValue(std::string k);
    KeyValue(KeyValue const& other);
    KeyValue(KeyValue&& other) noexcept;
    ~KeyValue();

    KeyValue& operator=(KeyValue&& other) noexcept = delete;

    void swap(KeyValue& other) {
        m_key_.swap(other.m_key_);
        m_node_.swap(other.m_node_);
    }
    KeyValue& operator=(KeyValue const& other);
    KeyValue& operator=(std::initializer_list<KeyValue> const& u);
    KeyValue& operator=(std::initializer_list<std::initializer_list<KeyValue>> const& other);
    KeyValue& operator=(std::initializer_list<std::initializer_list<std::initializer_list<KeyValue>>> const& u);

    template <typename U>
    KeyValue& operator=(U const& u) {
        m_node_ = DataNode::New(DataLight::New(u));
        return *this;
    }

    template <typename U>
    KeyValue& operator=(std::initializer_list<U> const& u) {
        m_node_ = DataNode::New(DataLight::New(u));
        return *this;
    }

    template <typename U>
    KeyValue& operator=(std::initializer_list<std::initializer_list<U>> const& u) {
        m_node_ = DataNode::New(DataLight::New(u));
        return *this;
    }
    template <typename U>
    KeyValue& operator=(std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& u) {
        m_node_ = DataNode::New(DataLight::New(u));
        return *this;
    }
};

inline KeyValue operator"" _(const char* c, std::size_t n) { return KeyValue(std::string(c)); }

#define SP_DATA_NODE_HEAD(_CLASS_NAME_, _BASE_NAME_)                                                                 \
                                                                                                                     \
   public:                                                                                                           \
    static std::string GetFancyTypeName_s() {                                                                        \
        return _BASE_NAME_::GetFancyTypeName_s() + "." + __STRING(_CLASS_NAME_);                                     \
    }                                                                                                                \
    virtual std::string GetFancyTypeName() const override { return GetFancyTypeName_s(); }                           \
    static bool _is_registered;                                                                                      \
                                                                                                                     \
   private:                                                                                                          \
    typedef _BASE_NAME_ base_type;                                                                                   \
    typedef _CLASS_NAME_ this_type;                                                                                  \
                                                                                                                     \
   public:                                                                                                           \
    explicit _CLASS_NAME_(_CLASS_NAME_ const& other) = delete;                                                       \
    explicit _CLASS_NAME_(_CLASS_NAME_&& other) = delete;                                                            \
    _CLASS_NAME_& operator=(_CLASS_NAME_ const& other) = delete;                                                     \
    _CLASS_NAME_& operator=(_CLASS_NAME_&& other) = delete;                                                          \
                                                                                                                     \
   protected:                                                                                                        \
    explicit _CLASS_NAME_(DataNode::eNodeType etype = DN_TABLE);                                                     \
                                                                                                                     \
   public:                                                                                                           \
    ~_CLASS_NAME_() override;                                                                                        \
                                                                                                                     \
   public:                                                                                                           \
    template <typename... Args>                                                                                      \
    static std::shared_ptr<this_type> New(Args&&... args) {                                                          \
        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...));                               \
    }                                                                                                                \
                                                                                                                     \
    std::shared_ptr<_CLASS_NAME_> Self() { return std::dynamic_pointer_cast<this_type>(this->shared_from_this()); }; \
    std::shared_ptr<_CLASS_NAME_> Self() const {                                                                     \
        return std::dynamic_pointer_cast<this_type>(const_cast<this_type*>(this)->shared_from_this());               \
    };                                                                                                               \
    std::shared_ptr<_CLASS_NAME_> Root() const {                                                                     \
        return std::dynamic_pointer_cast<this_type>(const_cast<this_type*>(this)->GetRoot());                        \
    };                                                                                                               \
    std::shared_ptr<_CLASS_NAME_> Parent() const {                                                                   \
        return std::dynamic_pointer_cast<this_type>(const_cast<this_type*>(this)->GetParent());                      \
    };

#define SP_DATA_NODE_FUNCTION(_CLASS_NAME_)                                                                          \
                                                                                                                     \
    std::shared_ptr<DataNode> CreateNode(eNodeType e_type) const override;                                           \
                                                                                                                     \
    size_type size() const override;                                                                                 \
                                                                                                                     \
    size_type Set(std::string const& uri, std::shared_ptr<DataNode> const& v) override;                              \
    size_type Add(std::string const& uri, std::shared_ptr<DataNode> const& v) override;                              \
    size_type Delete(std::string const& s) override;                                                                 \
    std::shared_ptr<DataNode> Get(std::string const& uri) const override;                                            \
    void Foreach(std::function<void(std::string const&, std::shared_ptr<DataNode> const&)> const& f) const override; \
                                                                                                                     \
    size_type Set(index_type s, std::shared_ptr<DataNode> const& v) override;                                        \
    size_type Add(index_type s, std::shared_ptr<DataNode> const& v) override;                                        \
    size_type Delete(index_type s) override;                                                                         \
    std::shared_ptr<DataNode> Get(index_type s) const override;

namespace detail {
template <typename U>
size_type _CopyFromData(U& dst, std::shared_ptr<DataNode> const& src, ENABLE_IF(std::rank<U>::value == 0)) {
    static auto snan = std::numeric_limits<U>::signaling_NaN();

    if (src == nullptr) {
        dst = snan;
        return 0;
    }

    size_type count = 0;
    switch (src->type()) {
        case DataNode::DN_ENTITY: {
            if (auto p = std::dynamic_pointer_cast<DataLightT<U>>(src->GetEntity())) {
                dst = p->value();
                count = 1;
            }
        } break;
        case DataNode::DN_NULL:
            break;
        case DataNode::DN_ARRAY:
        case DataNode::DN_TABLE:
        case DataNode::DN_FUNCTION:
        default:
            RUNTIME_ERROR << "Can not convert array/table/function to Single value!";
            break;
    }
    if (count == 0) { dst = std::numeric_limits<U>::signaling_NaN(); }
    return count;
};

template <typename U>
size_type _CopyFromData(U& dst, std::shared_ptr<DataNode> const& src, ENABLE_IF((std::rank<U>::value > 0))) {
    static auto snan = std::numeric_limits<traits::value_type_t<U>>::signaling_NaN();
    if (src == nullptr) {
        dst = snan;
        return 0;
    }
    size_type count = 0;
    switch (src->type()) {
        case DataNode::DN_TABLE:
        case DataNode::DN_ARRAY: {
            for (size_type i = 0; i < std::extent<U, 0>::value; ++i) { count += _CopyFromData(dst[i], src->Get(i)); }
        } break;
        case DataNode::DN_ENTITY: {
            if (auto p = std::dynamic_pointer_cast<DataLightT<traits::value_type_t<U>*>>(src->GetEntity())) {
                count = p->CopyOut(dst);
            }
        } break;
        case DataNode::DN_NULL:
        case DataNode::DN_FUNCTION:
        default:
            dst = std::numeric_limits<traits::value_type_t<U>>::signaling_NaN();
            break;
            //            RUNTIME_ERROR << "Can not convert entity/table/function to nTuple[" << std::rank<U>::value <<
            //            "]!";
            //            break;
    }
    if (count == 0) { dst = snan; };

    return count;
};

template <typename U>
size_type _CopyToData(std::shared_ptr<DataNode> dst, U const& src, ENABLE_IF(std::rank<U>::value == 1)) {
    if (dst == nullptr) { return 0; }
    static auto snan = std::numeric_limits<U>::signaling_NaN();
    size_type count = 0;
    switch (src->type()) {
        break;
        case DataNode::DN_ARRAY: {
            if (dst->size() > 0) {
                for (size_type i = 0; i < dst->size(); ++i) { count += dst->Set(i, src[i]); }
            }
        }
        case DataNode::DN_ENTITY:
        case DataNode::DN_TABLE:
        case DataNode::DN_FUNCTION:
        default:
            RUNTIME_ERROR << "Can not set single value to table/function!";
            break;
    }

    return count;
};

template <typename U>
size_type _CopyToData(std::shared_ptr<DataNode> dst, U const& src, ENABLE_IF((std::rank<U>::value > 0))) {
    if (dst == nullptr) { return 0; }
    size_type count = 0;
    static auto snan = std::numeric_limits<traits::value_type_t<U>>::signaling_NaN();

    switch (src->type()) {
        case DataNode::DN_ARRAY: {
            for (size_type i = 0, ie = std::min(std::extent<U, 0>::value, dst->size()); i < ie; ++i) {
                count += _CopyToData(dst->Get(i), src[i]);
            }
            for (size_type i = std::extent<U, 0>::value, ie = dst->size(); i < ie; ++i) {
                count += _CopyToData(dst->Get(i), snan);
            }
        } break;
        case DataNode::DN_ENTITY:
        case DataNode::DN_TABLE:
        case DataNode::DN_FUNCTION:
        default:
            RUNTIME_ERROR << "Can not convert entity/table/function to nTuple[" << std::rank<U>::value << "]!";
            break;
    }
    return count;
};

template <typename U>
bool _CompareToData(U const& dst, std::shared_ptr<DataNode> const& src, ENABLE_IF(std::rank<U>::value == 0)) {
    if (src == nullptr) { return false; }
    bool res;
    switch (src->type()) {
        case DataNode::DN_ENTITY: {
            if (auto p = std::dynamic_pointer_cast<DataLightT<U>>(src->GetEntity())) {
                res = (p->value() == dst);
            } else {
                res = false;
            }
        } break;
        case DataNode::DN_ARRAY:
        case DataNode::DN_TABLE:
        case DataNode::DN_FUNCTION:
        default:
            res = false;
            break;
    }

    return res;
};

template <typename U>
bool _CompareToData(U const& dst, std::shared_ptr<DataNode> const& src, ENABLE_IF((std::rank<U>::value > 0))) {
    if (src == nullptr) { return 0; }
    bool res = true;
    static auto snan = std::numeric_limits<traits::value_type_t<U>>::signaling_NaN();

    switch (src->type()) {
        case DataNode::DN_ARRAY: {
            if (std::extent<U, 0>::value != dst->size()) {
                res = false;
            } else {
                for (size_type i = 0, ie = std::min(std::extent<U, 0>::value, dst->size()); i < ie; ++i) {
                    res = res && _CompareToData(dst[i], src->Get(i));
                }
            }

        } break;
        case DataNode::DN_ENTITY:
        case DataNode::DN_TABLE:
        case DataNode::DN_FUNCTION:
        default:
            res = false;
            break;
    }
    return res;
};
}  // namespace detail
template <typename U>
size_type CopyFromData(U& dst, std::shared_ptr<DataNode> const& src) {
    return detail::_CopyFromData(dst, src);
}
template <typename U>
size_type CopyToData(std::shared_ptr<DataNode> dst, U const& src) {
    return detail::_CopyToData(dst, src);
}
template <typename U>
bool EqualTo(U const& dst, std::shared_ptr<DataNode> const& src) {
    return detail::_CompareToData(dst, src);
}

//
//    template <typename U>
//    size_type _CopyIn(std::shared_ptr<DataNode>& dst, U const& src, ENABLE_IF(std::rank<U>::value == 0)) const {
//        return dst == nullptr ? 0 : dst->SetValue(s, src);
//    };
//
//    template <typename U>
//    size_type _CopyIn(std::shared_ptr<DataNode>& dst, U const& src, ENABLE_IF((std::rank<U>::value > 0))) const {
//        size_type count = 0;
//        if (dst != nullptr) {
//            for (size_type i = 0; i < std::extent<U, 0>::value; ++i) {
//                //                count += _CopyIn(dst == nullptr ? dst : dst->Get(i, NEW_IF_NOT_EXIST), src[i]);
//            }
//        }
//        return count;
//    };
//    template <typename U>
//    static bool _isEqualTo(U const& left, std::shared_ptr<DataNode> const& right,
//                           ENABLE_IF(std::rank<U>::value == 0)) {
//        return right != nullptr && right->GetEntity() != nullptr;
//    };
//
//    template <typename U>
//    static bool _isEqualTo(U const& left, std::shared_ptr<DataNode> const& right,
//                           ENABLE_IF((std::rank<U>::value > 0))) {
//        bool res = true;
//        for (size_type i = 0; res && i < std::extent<U, 0>::value; ++i) {
//            res = res && right != nullptr && _isEqualTo(left[i], right->Get(i));
//        }
//        return res;
//    };
}  // namespace data
}  // namespace simpla

#endif  // SIMPLA_DATAENTRY_H
