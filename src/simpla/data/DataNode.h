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
size_type CopyFromData(U& dst, std::shared_ptr<const DataNode> const& src);
template <typename U>
size_type CopyToData(std::shared_ptr<DataNode> dst, U const& src);
template <typename U>
bool EqualTo(U const& dst, std::shared_ptr<const DataNode> const& src);

class DataNode : public Factory<DataNode>, public std::enable_shared_from_this<DataNode> {
    SP_DEFINE_FANCY_TYPE_NAME(DataNode, Factory<DataNode>);

   public:
    enum eNodeType { DN_NULL = 0, DN_ENTITY = 1, DN_TABLE = 2, DN_ARRAY = 3, DN_FUNCTION = 4 };

   protected:
    //    static int s_num_of_pre_registered_;
    const eNodeType m_type_;

    explicit DataNode(eNodeType etype = DN_TABLE);

   public:
    ~DataNode() override;

    static std::shared_ptr<DataNode> New(eNodeType e_type = DN_TABLE, std::string const& uri = "");
    eNodeType type() const;

    std::shared_ptr<DataNode> CreateEntity(std::shared_ptr<DataEntity> const&) const;
    std::shared_ptr<DataNode> GetRoot() const {
        return GetParent() == nullptr ? const_cast<DataNode*>(this)->shared_from_this() : GetParent();
    }
    void SetParent(std::shared_ptr<DataNode>) {}
    std::shared_ptr<DataNode> GetParent() const { return nullptr; }
    /** @addtogroup required @{*/
    virtual std::shared_ptr<DataNode> CreateNode(eNodeType e_type) const;

    virtual size_type size() const;

    virtual std::shared_ptr<DataEntity> GetEntity() const;
    virtual size_type SetEntity(const std::shared_ptr<DataEntity>&);

    virtual size_type Set(std::string const& uri, std::shared_ptr<DataNode> const& v);
    virtual size_type Add(std::string const& uri, std::shared_ptr<DataNode> const& v);
    virtual size_type Delete(std::string const& s);
    virtual std::shared_ptr<DataNode> Get(std::string const& uri) const;
    virtual size_type Foreach(std::function<size_type(std::string, std::shared_ptr<DataNode>)> const& f) const;

    virtual size_type Set(size_type s, std::shared_ptr<DataNode> const& v);
    virtual size_type Add(size_type s, std::shared_ptr<DataNode> const& v);
    virtual size_type Delete(size_type s);
    virtual std::shared_ptr<DataNode> Get(size_type s) const;

    virtual size_type Add(std::shared_ptr<DataNode> const& v);

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

    template <typename U>
    size_type SetValue(std::string const& s, U const& u, ENABLE_IF(traits::is_light_data<U>::value)) {
        return Set(s, CreateEntity(DataLight::New(u)));
    };
    size_type SetValue(std::string const& s, char const* u) {
        return Set(s, CreateEntity(DataLight::New(std::string(u))));
    };
    template <typename U>
    size_type SetValue(std::string const& s, U const& u, ENABLE_IF(!traits::is_light_data<U>::value)) {
        return Set(s, CreateEntity(DataBlock::New(u)));
    };
    template <typename U>
    size_type SetValue(std::string const& s, std::initializer_list<U> const& v) {
        return Set(s, CreateEntity(DataLight::New(v)));
    }

    template <typename U>
    size_type SetValue(std::string const& s, std::initializer_list<std::initializer_list<U>> const& v) {
        return Set(s, CreateEntity(DataLight::New(v)));
    }
    template <typename U>
    size_type SetValue(std::string const& s,
                       std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& v) {
        return Set(s, CreateEntity(DataLight::New(v)));
    }

    template <typename U>
    size_type AddValue(std::string const& s, U const& u, ENABLE_IF(traits::is_light_data<U>::value)) {
        return Add(s, CreateEntity(DataLight::New(u)));
    };
    template <typename U>
    size_type AddValue(std::string const& s, U const& u, ENABLE_IF(!traits::is_light_data<U>::value)) {
        return Add(s, CreateEntity(DataBlock::New(u)));
    };
    template <typename U>
    size_type AddValue(std::string const& s, std::initializer_list<U> const& v) {
        return Add(s, CreateEntity(DataLight::New(v)));
    }
    template <typename U>
    size_type AddValue(std::string const& s, std::initializer_list<std::initializer_list<U>> const& v) {
        return Add(s, CreateEntity(DataLight::New(v)));
    }
    template <typename U>
    size_type AddValue(std::string const& s,
                       std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& v) {
        return Add(s, CreateEntity(DataLight::New(v)));
    }
    template <typename U>
    U GetValue(std::string const& url) const {
        U res;
        if (CopyFromData(res, Get(url)) == 0) { res = std::numeric_limits<U>::signaling_NaN(); }
        return res;
    };

    template <typename U>
    U GetValue(std::string const& url, U default_value) const {
        CopyFromData(default_value, Get(url));
        return std::move(default_value);
    };
    std::string GetValue(std::string const& url, char const* default_value) const {
        std::string res(default_value);
        CopyFromData(res, Get(url));
        return std::move(res);
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
    //   public:
    //    template <typename U>
    //    size_type CopyOut(U& res) const {
    //        return CopyFromData(res, shared_from_this());
    //    };
    //    template <typename U>
    //    size_type CopyIn(U const& dst) {
    //        return CopyToData(shared_from_this(), dst);
    //    };
    //    template <typename U>
    //    bool isEqualTo(U const& dst) const {
    //        return EqualTo(dst, shared_from_this());
    //    };

    //******************************************************************************************************
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

    KeyValue(std::string k);
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
        m_node_ = DataNode::New(DataNode::DN_ENTITY, "");
        m_node_->SetEntity(DataLight::New(u));
        return *this;
    }

    template <typename U>
    KeyValue& operator=(std::initializer_list<U> const& u) {
        m_node_ = DataNode::New(DataNode::DN_ENTITY, "");
        m_node_->SetEntity(DataLight::New(u));
        return *this;
    }

    template <typename U>
    KeyValue& operator=(std::initializer_list<std::initializer_list<U>> const& u) {
        m_node_ = DataNode::New(DataNode::DN_ENTITY, "");
        m_node_->SetEntity(DataLight::New(u));
        return *this;
    }
    template <typename U>
    KeyValue& operator=(std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& u) {
        m_node_ = DataNode::New(DataNode::DN_ENTITY, "");
        m_node_->SetEntity(DataLight::New(u));
        return *this;
    }
};

inline KeyValue operator"" _(const char* c, std::size_t n) { return KeyValue(std::string(c)); }

#define SP_DATA_NODE_HEAD(_CLASS_NAME_)                                                                              \
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

#define SP_DATA_NODE_FUNCTION(_CLASS_NAME_)                                                                      \
   protected:                                                                                                    \
    explicit _CLASS_NAME_(DataNode::eNodeType etype = DataNode::DN_TABLE);                                       \
                                                                                                                 \
   public:                                                                                                       \
    explicit _CLASS_NAME_(_CLASS_NAME_ const& other) = delete;                                                   \
    explicit _CLASS_NAME_(_CLASS_NAME_&& other) = delete;                                                        \
    _CLASS_NAME_& operator=(_CLASS_NAME_ const& other) = delete;                                                 \
    _CLASS_NAME_& operator=(_CLASS_NAME_&& other) = delete;                                                      \
                                                                                                                 \
    ~_CLASS_NAME_() override;                                                                                    \
                                                                                                                 \
    std::shared_ptr<DataNode> CreateNode(eNodeType e_type) const override;                                       \
                                                                                                                 \
    size_type size() const override;                                                                             \
                                                                                                                 \
    std::shared_ptr<DataEntity> GetEntity() const override;                                                      \
    size_type SetEntity(const std::shared_ptr<DataEntity>&) override;                                            \
                                                                                                                 \
    size_type Set(std::string const& uri, std::shared_ptr<DataNode> const& v) override;                          \
    size_type Add(std::string const& uri, std::shared_ptr<DataNode> const& v) override;                          \
    size_type Delete(std::string const& s) override;                                                             \
    std::shared_ptr<DataNode> Get(std::string const& uri) const override;                                        \
    size_type Foreach(std::function<size_type(std::string, std::shared_ptr<DataNode>)> const& f) const override; \
                                                                                                                 \
    size_type Set(size_type s, std::shared_ptr<DataNode> const& v) override;                                     \
    size_type Add(size_type s, std::shared_ptr<DataNode> const& v) override;                                     \
    size_type Delete(size_type s) override;                                                                      \
    std::shared_ptr<DataNode> Get(size_type s) const override;

namespace detail {
template <typename U>
size_type _CopyFromData(U& dst, std::shared_ptr<const DataNode> const& src, ENABLE_IF(std::rank<U>::value == 0)) {
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
size_type _CopyFromData(U& dst, std::shared_ptr<const DataNode> const& src, ENABLE_IF((std::rank<U>::value > 0))) {
    static auto snan = std::numeric_limits<traits::value_type_t<U>>::signaling_NaN();
    if (src == nullptr) {
        dst = snan;
        return 0;
    }
    size_type count = 0;
    switch (src->type()) {
        case DataNode::DN_ARRAY: {
            for (size_type i = 0; i < std::extent<U, 0>::value; ++i) { count += _CopyFromData(dst[i], src->Get(i)); }
        } break;
        case DataNode::DN_ENTITY: {
            if (auto p = std::dynamic_pointer_cast<DataLightT<traits::value_type_t<U>*>>(src->GetEntity())) {
                count = p->CopyOut(dst);
            }
        } break;
        case DataNode::DN_NULL:
        case DataNode::DN_TABLE:
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
size_type _CopyToData(std::shared_ptr<DataNode> dst, U const& src, ENABLE_IF(std::rank<U>::value == 0)) {
    if (dst == nullptr) { return 0; }
    static auto snan = std::numeric_limits<U>::signaling_NaN();
    size_type count = 0;
    switch (src->type()) {
        case DataNode::DN_ENTITY: {
            count = dst->SetEntity(DataLightT<U>::New(src));
        } break;
        case DataNode::DN_ARRAY: {
            if (dst->size() > 0) {
                _CopyToData(dst->Get(0), snan);
                for (size_type i = 1; i < dst->size(); ++i) { count += _CopyToData(dst->Get(i), src); }
            }
        }
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
bool _CompareToData(U const& dst, std::shared_ptr<const DataNode> const& src, ENABLE_IF(std::rank<U>::value == 0)) {
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
bool _CompareToData(U const& dst, std::shared_ptr<const DataNode> const& src, ENABLE_IF((std::rank<U>::value > 0))) {
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
size_type CopyFromData(U& dst, std::shared_ptr<const DataNode> const& src) {
    return detail::_CopyFromData(dst, src);
}
template <typename U>
size_type CopyToData(std::shared_ptr<DataNode> dst, U const& src) {
    return detail::_CopyToData(dst, src);
}
template <typename U>
bool EqualTo(U const& dst, std::shared_ptr<const DataNode> const& src) {
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
//    static bool _isEqualTo(U const& left, std::shared_ptr<const DataNode> const& right,
//                           ENABLE_IF(std::rank<U>::value == 0)) {
//        return right != nullptr && right->GetEntity() != nullptr;
//    };
//
//    template <typename U>
//    static bool _isEqualTo(U const& left, std::shared_ptr<const DataNode> const& right,
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
