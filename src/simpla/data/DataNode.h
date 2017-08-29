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
class ConstKeyValue;

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
    std::shared_ptr<DataEntity> GetEntity(std::string const& url) const {
        std::shared_ptr<DataEntity> res = nullptr;
        if (auto p = Get(url)) { res = p->GetEntity(); }
        return res;
    };
    size_type Set(std::string const& uri, std::shared_ptr<const DataNode> const& v);
    size_type Add(std::string const& uri, std::shared_ptr<const DataNode> const& v);

    ConstKeyValue operator[](std::string const& s) const;
    ConstKeyValue operator[](index_type s) const;
    KeyValue operator[](std::string const& s);
    KeyValue operator[](index_type s);

    template <typename TI, typename U>
    size_type SetValue(TI const& s, U const& u, ENABLE_IF(traits::is_light_data<U>::value)) {
        return Set(s, DataLight::New(u));
    };
    template <typename TI>
    size_type SetValue(TI const& s, char const* u) {
        return Set(s, DataLight::New(std::string(u)));
    };
    template <typename TI, typename U>
    size_type SetValue(TI const& s, U const& u, ENABLE_IF(!traits::is_light_data<U>::value)) {
        return Set(s, DataBlock::New(u));
    };
    template <typename TI, typename U>
    size_type SetValue(TI const& s, std::initializer_list<U> const& v) {
        return Set(s, DataLight::New(v));
    }

    template <typename TI, typename U>
    size_type SetValue(TI const& s, std::initializer_list<std::initializer_list<U>> const& v) {
        return Set(s, DataLight::New(v));
    }
    template <typename TI, typename U>
    size_type SetValue(TI const& s, std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& v) {
        return Set(s, DataLight::New(v));
    }

    template <typename TI, typename U>
    size_type AddValue(TI const& s, U const& u, ENABLE_IF(traits::is_light_data<U>::value)) {
        return Add(s, DataLight::New(u));
    };
    template <typename TI, typename U>
    size_type AddValue(TI const& s, U const& u, ENABLE_IF(!traits::is_light_data<U>::value)) {
        return Add(s, DataBlock::New(u));
    };
    template <typename TI, typename U>
    size_type AddValue(TI const& s, std::initializer_list<U> const& v) {
        return Add(s, DataLight::New(v));
    }
    template <typename TI, typename U>
    size_type AddValue(TI const& s, std::initializer_list<std::initializer_list<U>> const& v) {
        return Add(s, DataLight::New(v));
    }
    template <typename TI, typename U>
    size_type AddValue(TI const& s, std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& v) {
        return Add(s, DataLight::New(v));
    }
    template <typename U>
    U GetValue(std::string const& url, ENABLE_IF((traits::is_light_data<U>::value))) const {
        U res;
        size_type count = 0;
        if (auto p = Get(url)) { count = p->CopyOut(res); }
        if (count == 0) { res = std::numeric_limits<U>::signaling_NaN(); }
        return res;
    };

    template <typename U>
    std::shared_ptr<DataBlockT<traits::value_type_t<U>>> GetValue(ENABLE_IF((!traits::is_light_data<U>::value))) const {
        return std::dynamic_pointer_cast<DataBlockT<traits::value_type_t<U>>>(GetEntity());
    }

    template <typename U>
    U GetValue(std::string const& url, U const& default_value) const {
        U res;
        size_type count = 0;
        if (auto p = Get(url)) { count = p->CopyOut(res); }
        return count > 0 ? res : default_value;
    };

    std::shared_ptr<DataBlock> GetData() const { return std::dynamic_pointer_cast<DataBlock>(GetEntity()); }

    template <typename URL, typename U>
    bool Check(URL const& url, U const& u) const {
        auto p = Get(url);
        return p != nullptr && p->isEqualTo(u);
    }

    bool Check(std::string const& uri) const { return Check(uri, true); }

    /** @} */
    //
    //    //******************************************************************************************************
   private:
    template <typename U>
    size_type _CopyOut(U& dst, std::shared_ptr<const DataNode> const& src, ENABLE_IF(std::rank<U>::value == 0)) const {
        size_type count = 0;
        if (src == nullptr) {
            dst = std::numeric_limits<U>::signaling_NaN();
        } else if (auto p = std::dynamic_pointer_cast<DataLightT<U>>(src->GetEntity())) {
            dst = p->value();

        } else {
            dst = std::numeric_limits<U>::signaling_NaN();
        }
        count = 1;
        return count;
    };

    template <typename U>
    size_type _CopyOut(U& dst, std::shared_ptr<const DataNode> const& src, ENABLE_IF((std::rank<U>::value > 0))) const {
        size_type count = 0;
        if (src == nullptr) {
            dst = std::numeric_limits<U>::signaling_NaN();
        } else if (auto p = std::dynamic_pointer_cast<DataLightT<traits::value_type_t<U>*>>(src->GetEntity())) {
            count = p->CopyOut(dst);
        } else {
            for (size_type i = 0; i < std::extent<U, 0>::value; ++i) { count += _CopyOut(dst[i], src->Get(i)); }
        }
        return count;
    };
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

   public:
    template <typename U>
    size_type CopyOut(U& res) const {
        return _CopyOut(res, shared_from_this());
    };
    template <typename U>
    size_type CopyIn(U const& dst){
        //        return _CopyIn(dst, shared_from_this());
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
            //            res = _isEqualTo(dst, shared_from_this());
        }
        return res;
    };

    //******************************************************************************************************
};
std::pair<std::string, std::shared_ptr<DataNode>> RecursiveFindNode(std::shared_ptr<DataNode> d, std::string uri,
                                                                    int flag = 0);

std::ostream& operator<<(std::ostream&, DataNode const&);

struct KeyValue {
    std::string m_key_;
    std::shared_ptr<DataNode> m_parent_;

    explicit KeyValue(std::string const& k, std::shared_ptr<DataNode> const& p = nullptr)
        : m_key_(k), m_parent_(p == nullptr ? DataNode::New() : p) {}
    template <typename U>
    KeyValue(std::string const& k, U const& u) : m_key_((k)), m_parent_(DataNode::New()) {
        this->operator=(u);
    }

    KeyValue(KeyValue const& other) : m_key_(other.m_key_), m_parent_(other.m_parent_) {}
    KeyValue(KeyValue&& other) noexcept : m_key_(other.m_key_), m_parent_(other.m_parent_) {}

    ~KeyValue() = default;

    KeyValue& operator=(KeyValue&& other) noexcept = delete;

    void swap(KeyValue& other) {
        m_key_.swap(other.m_key_);
        m_parent_.swap(other.m_parent_);
    }

    std::shared_ptr<const DataNode> operator->() const {
        return m_parent_ == nullptr ? nullptr : m_parent_->Get(m_key_);
    }

    template <typename U>
    U as() const {
        return m_parent_->GetValue<U>(m_key_);
    }
    template <typename U>
    operator U() const {
        return as<U>();
    }
    template <typename U>
    U as(U const& default_value) const {
        return m_parent_->GetValue<U>(m_key_, default_value);
    }
    template <typename U>
    bool operator==(U const& other) const {
        return m_parent_->Check(m_key_, other);
    }

    template <typename U>
    KeyValue& operator=(U const& u) {
        m_parent_->SetValue(m_key_, u);
        return *this;
    }

    template <typename U>
    KeyValue& operator=(std::initializer_list<U> const& u) {
        m_parent_->SetValue(m_key_, u);
        return *this;
    }

    template <typename U>
    KeyValue& operator=(std::initializer_list<std::initializer_list<U>> const& u) {
        m_parent_->SetValue(m_key_, u);
        return *this;
    }
    template <typename U>
    KeyValue& operator=(std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& u) {
        m_parent_->SetValue(m_key_, u);
        return *this;
    }

    template <typename U>
    KeyValue& operator+=(U const& u) {
        m_parent_->AddValue(m_key_, u);
        return *this;
    }

    template <typename U>
    KeyValue& operator+=(std::initializer_list<U> const& u) {
        m_parent_->AddValue(m_key_, u);
        return *this;
    }

    template <typename U>
    KeyValue& operator+=(std::initializer_list<std::initializer_list<U>> const& u) {
        m_parent_->AddValue(m_key_, u);
        return *this;
    }
    template <typename U>
    KeyValue& operator+=(std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& u) {
        m_parent_->AddValue(m_key_, u);
        return *this;
    }

    KeyValue& operator=(KeyValue const& other) {
        m_parent_->Set(m_key_, other.m_parent_);
        return *this;
    }
    KeyValue& operator=(std::initializer_list<KeyValue> const& u) {
        for (auto const& v : u) { m_parent_->Set(m_key_, v.m_parent_); }
        return *this;
    }
};

struct ConstKeyValue {
    std::string m_key_;
    std::shared_ptr<const DataNode> m_parent_ = nullptr;

    explicit ConstKeyValue(std::string k, std::shared_ptr<const DataNode> const& p = nullptr)
        : m_key_(std::move(k)), m_parent_(p) {}
    ConstKeyValue(ConstKeyValue const& other) : m_key_(other.m_key_), m_parent_(other.m_parent_) {}
    ConstKeyValue(ConstKeyValue&& other) noexcept : m_key_(other.m_key_), m_parent_(other.m_parent_) {}
    ~ConstKeyValue() = default;
    ConstKeyValue& operator=(ConstKeyValue const& other) {
        ConstKeyValue(other).swap(*this);
        return *this;
    }
    ConstKeyValue& operator=(ConstKeyValue&& other) noexcept {
        ConstKeyValue(other).swap(*this);
        return *this;
    }
    void swap(ConstKeyValue& other) {
        m_key_.swap(other.m_key_);
        m_parent_.swap(other.m_parent_);
    }

    std::shared_ptr<const DataNode> operator->() const {
        return m_parent_ == nullptr ? nullptr : m_parent_->Get(m_key_);
    }

    template <typename U>
    U as() const {
        return m_parent_->GetValue<U>(m_key_);
    }
    template <typename U>
    explicit operator U() const {
        return as<U>();
    }
    template <typename U>
    U as(U const& default_value) const {
        return m_parent_->GetValue<U>(m_key_, default_value);
    }

    template <typename U>
    bool operator==(U const& other) const {
        return m_parent_->Check(m_key_, other);
    }
};
inline KeyValue operator"" _(const char* c, std::size_t n) { return KeyValue(std::string(c), true); }

inline ConstKeyValue DataNode::operator[](std::string const& s) const { return ConstKeyValue(s, shared_from_this()); }
inline ConstKeyValue DataNode::operator[](index_type s) const {
    return ConstKeyValue(std::to_string(s), shared_from_this());
}
inline KeyValue DataNode::operator[](std::string const& s) { return KeyValue(s, shared_from_this()); };
inline KeyValue DataNode::operator[](index_type s) { return KeyValue(std::to_string(s), shared_from_this()); }

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
    size_type Foreach(std::function<size_type(std::string, std::shared_ptr<const DataNode>)> const& f) const override;

}  // namespace data
}  // namespace simpla

#endif  // SIMPLA_DATAENTRY_H
