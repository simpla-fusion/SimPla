//
// Created by salmon on 17-8-18.
//

#ifndef SIMPLA_DATAENTRY_H
#define SIMPLA_DATAENTRY_H

#include "simpla/SIMPLA_config.h"

#include <simpla/utilities/Factory.h>
#include <simpla/utilities/Log.h>
#include <simpla/utilities/ObjectHead.h>
#include <memory>
#include "DataLight.h"
#include "DataUtilities.h"
namespace simpla {
namespace data {
#define SP_URL_SPLIT_CHAR '/'
class DataEntity;
class KeyValue;
class DataEntry;
template <typename U>
size_type CopyFromData(U& dst, std::shared_ptr<const DataEntry> const& src);
template <typename U>
size_type CopyToData(std::shared_ptr<DataEntry> dst, U const& src);
template <typename U>
bool EqualTo(U const& dst, std::shared_ptr<const DataEntry> const& src);

class DataEntry : public std::enable_shared_from_this<DataEntry> {
   public:
    virtual std::string FancyTypeName() const { return "DataEntity"; }
    enum eNodeType { DN_NULL = 0, DN_ENTITY = 1, DN_TABLE = 2, DN_ARRAY = 3, DN_FUNCTION = 4 };

   private:
    typedef DataEntry this_type;

   protected:
    typedef DataEntry base_type;

   protected:
    eNodeType m_type_ = DN_NULL;
    std::shared_ptr<DataEntity> m_entity_ = nullptr;
    std::shared_ptr<DataEntry> m_parent_ = nullptr;

   protected:
    explicit DataEntry(eNodeType etype = DN_TABLE);
    DataEntry(DataEntry const&);
    explicit DataEntry(std::shared_ptr<DataEntity> const&);
    explicit DataEntry(std::shared_ptr<const DataEntity> const&);

   public:
    virtual ~DataEntry();
    virtual std::shared_ptr<DataEntry> Copy() const;

    static std::shared_ptr<DataEntry> New(std::string const& uri = "");
    static std::shared_ptr<DataEntry> New(eNodeType e_type, std::string const& uri = "");
    static std::shared_ptr<DataEntry> New(std::shared_ptr<DataEntity> const& v);

    eNodeType type() const;
    std::shared_ptr<const DataEntry> GetRoot() const {
        return GetParent() == nullptr ? const_cast<DataEntry*>(this)->shared_from_this() : GetParent()->GetRoot();
    }
    std::shared_ptr<DataEntry> GetRoot() {
        return GetParent() == nullptr ? const_cast<DataEntry*>(this)->shared_from_this() : GetParent()->GetRoot();
    }
    bool isRoot() const { return GetParent() == nullptr; }
    void SetParent(std::shared_ptr<DataEntry> const& p) { m_parent_ = p; }
    std::shared_ptr<const DataEntry> GetParent() const { return m_parent_; }
    std::shared_ptr<DataEntry> GetParent() { return m_parent_; }

    /** @addtogroup required @{*/
    virtual std::shared_ptr<DataEntry> CreateNode(eNodeType e_type) const;
    virtual std::shared_ptr<DataEntry> CreateNode(std::string const& url, eNodeType e_type);

    virtual size_type size() const;
    virtual bool empty() const { return size() == 0; }

    std::shared_ptr<const DataEntity> GetEntity(int N) const;
    std::shared_ptr<DataEntity> GetEntity(int N);
    virtual std::shared_ptr<const DataEntity> GetEntity() const;
    virtual std::shared_ptr<DataEntity> GetEntity();
    virtual void SetEntity(std::shared_ptr<DataEntity> const& e);
    void SetEntity(std::shared_ptr<const DataEntity> const& e);

    virtual size_type Set(std::string const& uri, const std::shared_ptr<DataEntry>& v);
    virtual size_type Set(index_type s, const std::shared_ptr<DataEntry>& v);
    virtual size_type Set(const std::shared_ptr<DataEntry>& v);
    size_type Set(std::string const& uri, const std::shared_ptr<const DataEntry>& v);
    size_type Set(index_type s, const std::shared_ptr<const DataEntry>& v);
    size_type Set(const std::shared_ptr<const DataEntry>& v);

    virtual size_type Add(index_type s, const std::shared_ptr<DataEntry>& v);
    virtual size_type Add(std::string const& uri, const std::shared_ptr<DataEntry>& v);
    virtual size_type Add(const std::shared_ptr<DataEntry>& v);
    size_type Add(std::string const& uri, const std::shared_ptr<const DataEntry>& v);
    size_type Add(index_type s, const std::shared_ptr<const DataEntry>& v);
    size_type Add(const std::shared_ptr<const DataEntry>& v);

    virtual size_type Delete(std::string const& s);
    virtual std::shared_ptr<const DataEntry> Get(std::string const& uri) const;
    virtual std::shared_ptr<DataEntry> Get(std::string const& uri);

    virtual void Foreach(
        std::function<void(std::string const&, std::shared_ptr<const DataEntry> const&)> const& f) const;
    virtual void Foreach(std::function<void(std::string const&, std::shared_ptr<DataEntry> const&)> const& f);

    virtual size_type Delete(index_type s);
    virtual std::shared_ptr<const DataEntry> Get(index_type s) const;
    virtual std::shared_ptr<DataEntry> Get(index_type s);

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
        return Set(s, DataEntry::New(DataLight::New(std::forward<Args>(args)...)));
    };
    template <typename U>
    size_type SetValue(std::string const& s, U const& v) {
        return Set(s, DataEntry::New(DataLight::New(v)));
    }
    template <typename U>
    size_type SetValue(std::string const& s, std::initializer_list<U> const& v) {
        return Set(s, DataEntry::New(DataLight::New(v)));
    }
    template <typename U>
    size_type SetValue(std::string const& s, std::initializer_list<std::initializer_list<U>> const& v) {
        return Set(s, DataEntry::New(DataLight::New(v)));
    }
    template <typename U>
    size_type SetValue(std::string const& s,
                       std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& v) {
        return Set(s, DataEntry::New(DataLight::New(v)));
    }

    template <typename U>
    size_type AddValue(std::string const& s, U const& u) {
        return Add(s, DataEntry::New(DataLight::New(u)));
    };

    template <typename U>
    size_type AddValue(std::string const& s, std::initializer_list<U> const& v) {
        return Add(s, DataEntry::New(DataLight::New(v)));
    }
    template <typename U>
    size_type AddValue(std::string const& s, std::initializer_list<std::initializer_list<U>> const& v) {
        return Add(s, DataEntry::New(DataLight::New(v)));
    }
    template <typename U>
    size_type AddValue(std::string const& s,
                       std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& v) {
        return Add(s, DataEntry::New(DataLight::New(v)));
    }

    //    template <typename U>
    //    size_type SetValue(std::string const& s, U const& u, ENABLE_IF(!traits::is_light_data<U>::value)) {
    //        return Set(s, DataEntry::New(DataBlock::New(u)));
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

std::ostream& operator<<(std::ostream&, DataEntry const&);
std::istream& operator>>(std::istream&, DataEntry&);

struct KeyValue {
    std::string m_key_;
    std::shared_ptr<DataEntry> m_node_;

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
        m_node_ = DataEntry::New(DataLight::New(u));
        return *this;
    }

    template <typename U>
    KeyValue& operator=(std::initializer_list<U> const& u) {
        m_node_ = DataEntry::New(DataLight::New(u));
        return *this;
    }

    template <typename U>
    KeyValue& operator=(std::initializer_list<std::initializer_list<U>> const& u) {
        m_node_ = DataEntry::New(DataLight::New(u));
        return *this;
    }
    template <typename U>
    KeyValue& operator=(std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& u) {
        m_node_ = DataEntry::New(DataLight::New(u));
        return *this;
    }
};

inline KeyValue operator"" _(const char* c, std::size_t n) { return KeyValue(std::string(c)); }

#define SP_DATA_ENTITY_HEAD(_BASE_NAME_, _CLASS_NAME_, _REGISTER_NAME_)                                              \
   private:                                                                                                          \
    typedef _CLASS_NAME_ this_type;                                                                                  \
    typedef _BASE_NAME_ base_type;                                                                                   \
                                                                                                                     \
   public:                                                                                                           \
    std::string FancyTypeName() const override { return base_type::FancyTypeName() + "." + __STRING(_CLASS_NAME_); } \
                                                                                                                     \
   private:                                                                                                          \
    template <typename... Args>                                                                                      \
    static auto TryNew(std::true_type, Args&&... args) {                                                             \
        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...));                               \
    }                                                                                                                \
    template <typename... Args>                                                                                      \
    static auto TryNew(std::false_type, Args&&... args) {                                                            \
        RUNTIME_ERROR << __STRING(_CLASS_NAME_) << " is not constructible!";                                         \
        return nullptr;                                                                                              \
    }                                                                                                                \
                                                                                                                     \
   public:                                                                                                           \
    template <typename... Args>                                                                                      \
    static std::shared_ptr<this_type> New(Args&&... args) {                                                          \
        return TryNew(std::is_constructible<this_type, Args...>(), std::forward<Args>(args)...);                     \
    }                                                                                                                \
                                                                                                                     \
    std::shared_ptr<this_type> Self() { return std::dynamic_pointer_cast<this_type>(shared_from_this()); }           \
    std::shared_ptr<const this_type> Self() const {                                                                  \
        return std::dynamic_pointer_cast<const this_type>(shared_from_this());                                       \
    }                                                                                                                \
                                                                                                                     \
   private:                                                                                                          \
    static bool _is_registered;                                                                                      \
                                                                                                                     \
   public:                                                                                                           \
    static std::string RegisterName() { return __STRING(_REGISTER_NAME_); }                                          \
                                                                                                                     \
   public:                                                                                                           \
    _CLASS_NAME_(_CLASS_NAME_ const&);                                                                               \
    _CLASS_NAME_(data::DataEntry::eNodeType e_type = DN_ENTITY);                                                     \
                                                                                                                     \
    ~_CLASS_NAME_() override;                                                                                        \
                                                                                                                     \
    std::shared_ptr<_CLASS_NAME_> Root() const {                                                                     \
        return std::dynamic_pointer_cast<this_type>(const_cast<this_type*>(this)->GetRoot());                        \
    };                                                                                                               \
    std::shared_ptr<_CLASS_NAME_> Parent() const {                                                                   \
        return std::dynamic_pointer_cast<this_type>(const_cast<this_type*>(this)->GetParent());                      \
    };

//    using base_type::Set;                                                                                            \
//    using base_type::Add;                                                                                            \
//    using base_type::Get;                                                                                            \
//    size_type Set(std::string const& uri, std::shared_ptr<DataEntry> const& v) override;                             \
//    size_type Set(index_type s, std::shared_ptr<DataEntry> const& v) override;                                       \
//    size_type Add(std::string const& uri, std::shared_ptr<DataEntry> const& v) override;                             \
//    size_type Add(index_type s, std::shared_ptr<DataEntry> const& v) override;                                       \
//    size_type Delete(std::string const& s) override;                                                                 \
//    size_type Delete(index_type s) override;                                                                         \
//    std::shared_ptr<const DataEntry> Get(std::string const& uri) const override;                                     \
//    std::shared_ptr<const DataEntry> Get(index_type s) const override;                                               \
//    std::shared_ptr<DataEntry> Get(std::string const& uri) override;                                                 \
//    std::shared_ptr<DataEntry> Get(index_type s) override;                                                           \
//    void Foreach(std::function<void(std::string const&, std::shared_ptr<DataEntry> const&)> const& f) override;      \
//    void Foreach(std::function<void(std::string const&, std::shared_ptr<const DataEntry> const&)> const& f)          \
//        const override;

namespace detail {
template <typename U>
size_type _CopyFromData(U& dst, std::shared_ptr<const DataEntry> const& src, ENABLE_IF(std::rank<U>::value == 0)) {
    static auto snan = std::numeric_limits<U>::signaling_NaN();

    if (src == nullptr) {
        dst = snan;
        return 0;
    }

    size_type count = 0;
    switch (src->type()) {
        case DataEntry::DN_ENTITY: {
            if (auto p = std::dynamic_pointer_cast<const DataLightT<U>>(src->GetEntity())) {
                dst = p->value();
                count = 1;
            }
        } break;
        case DataEntry::DN_NULL:
            break;
        case DataEntry::DN_ARRAY:
        case DataEntry::DN_TABLE:
        case DataEntry::DN_FUNCTION:
        default:
            RUNTIME_ERROR << "Can not convert array/table/function to Single value!";
            break;
    }
    if (count == 0) { dst = std::numeric_limits<U>::signaling_NaN(); }
    return count;
};

template <typename U>
size_type _CopyFromData(U& dst, std::shared_ptr<const DataEntry> const& src, ENABLE_IF((std::rank<U>::value > 0))) {
    static auto snan = std::numeric_limits<traits::value_type_t<U>>::signaling_NaN();
    if (src == nullptr) {
        dst = snan;
        return 0;
    }
    size_type count = 0;
    switch (src->type()) {
        case DataEntry::DN_TABLE:
        case DataEntry::DN_ARRAY: {
            for (size_type i = 0; i < std::extent<U, 0>::value; ++i) { count += _CopyFromData(dst[i], src->Get(i)); }
        } break;
        case DataEntry::DN_ENTITY: {
            if (auto p = std::dynamic_pointer_cast<const DataLightT<traits::value_type_t<U>*>>(src->GetEntity())) {
                count = p->CopyOut(dst);
            }
        } break;
        case DataEntry::DN_NULL:
        case DataEntry::DN_FUNCTION:
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
size_type _CopyToData(std::shared_ptr<DataEntry> dst, U const& src, ENABLE_IF(std::rank<U>::value == 1)) {
    if (dst == nullptr) { return 0; }
    static auto snan = std::numeric_limits<U>::signaling_NaN();
    size_type count = 0;
    switch (src->type()) {
        break;
        case DataEntry::DN_ARRAY: {
            if (dst->size() > 0) {
                for (size_type i = 0; i < dst->size(); ++i) { count += dst->Set(i, src[i]); }
            }
        }
        case DataEntry::DN_ENTITY:
        case DataEntry::DN_TABLE:
        case DataEntry::DN_FUNCTION:
        default:
            RUNTIME_ERROR << "Can not set single value to table/function!";
            break;
    }

    return count;
};

template <typename U>
size_type _CopyToData(std::shared_ptr<DataEntry> dst, U const& src, ENABLE_IF((std::rank<U>::value > 0))) {
    if (dst == nullptr) { return 0; }
    size_type count = 0;
    static auto snan = std::numeric_limits<traits::value_type_t<U>>::signaling_NaN();

    switch (src->type()) {
        case DataEntry::DN_ARRAY: {
            for (size_type i = 0, ie = std::min(std::extent<U, 0>::value, dst->size()); i < ie; ++i) {
                count += _CopyToData(dst->Get(i), src[i]);
            }
            for (size_type i = std::extent<U, 0>::value, ie = dst->size(); i < ie; ++i) {
                count += _CopyToData(dst->Get(i), snan);
            }
        } break;
        case DataEntry::DN_ENTITY:
        case DataEntry::DN_TABLE:
        case DataEntry::DN_FUNCTION:
        default:
            RUNTIME_ERROR << "Can not convert entity/table/function to nTuple[" << std::rank<U>::value << "]!";
            break;
    }
    return count;
};

template <typename U>
bool _CompareToData(U const& dst, std::shared_ptr<const DataEntry> const& src, ENABLE_IF(std::rank<U>::value == 0)) {
    if (src == nullptr) { return false; }
    bool res;
    switch (src->type()) {
        case DataEntry::DN_ENTITY: {
            if (auto p = std::dynamic_pointer_cast<const DataLightT<U>>(src->GetEntity())) {
                res = (p->value() == dst);
            } else {
                res = false;
            }
        } break;
        case DataEntry::DN_ARRAY:
        case DataEntry::DN_TABLE:
        case DataEntry::DN_FUNCTION:
        default:
            res = false;
            break;
    }

    return res;
};

template <typename U>
bool _CompareToData(U const& dst, std::shared_ptr<DataEntry> const& src, ENABLE_IF((std::rank<U>::value > 0))) {
    if (src == nullptr) { return 0; }
    bool res = true;
    static auto snan = std::numeric_limits<traits::value_type_t<U>>::signaling_NaN();

    switch (src->type()) {
        case DataEntry::DN_ARRAY: {
            if (std::extent<U, 0>::value != dst->size()) {
                res = false;
            } else {
                for (size_type i = 0, ie = std::min(std::extent<U, 0>::value, dst->size()); i < ie; ++i) {
                    res = res && _CompareToData(dst[i], src->Get(i));
                }
            }

        } break;
        case DataEntry::DN_ENTITY:
        case DataEntry::DN_TABLE:
        case DataEntry::DN_FUNCTION:
        default:
            res = false;
            break;
    }
    return res;
};
}  // namespace detail
template <typename U>
size_type CopyFromData(U& dst, std::shared_ptr<const DataEntry> const& src) {
    return detail::_CopyFromData(dst, src);
}
template <typename U>
size_type CopyToData(std::shared_ptr<DataEntry> dst, U const& src) {
    return detail::_CopyToData(dst, src);
}
template <typename U>
bool EqualTo(U const& dst, std::shared_ptr<const DataEntry> const& src) {
    return detail::_CompareToData(dst, src);
}

//
//    template <typename U>
//    size_type _CopyIn(std::shared_ptr<DataEntry>& dst, U const& src, ENABLE_IF(std::rank<U>::value == 0)) const {
//        return dst == nullptr ? 0 : dst->SetValue(s, src);
//    };
//
//    template <typename U>
//    size_type _CopyIn(std::shared_ptr<DataEntry>& dst, U const& src, ENABLE_IF((std::rank<U>::value > 0))) const {
//        size_type count = 0;
//        if (dst != nullptr) {
//            for (size_type i = 0; i < std::extent<U, 0>::value; ++i) {
//                //                count += _CopyIn(dst == nullptr ? dst : dst->Get(i, NEW_IF_NOT_EXIST), src[i]);
//            }
//        }
//        return count;
//    };
//    template <typename U>
//    static bool _isEqualTo(U const& left, std::shared_ptr<DataEntry> const& right,
//                           ENABLE_IF(std::rank<U>::value == 0)) {
//        return right != nullptr && right->GetEntity() != nullptr;
//    };
//
//    template <typename U>
//    static bool _isEqualTo(U const& left, std::shared_ptr<DataEntry> const& right,
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
