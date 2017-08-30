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
   public:
    std::shared_ptr<DataNode> m_parent_ = nullptr;

    enum { RECURSIVE = 0b01, NEW_IF_NOT_EXIST = 0b010, ADD_IF_EXIST = 0b100, ONLY_TABLE = 0b1000 };
    enum eNodeType { DN_NULL = 0, DN_ENTITY = 1, DN_TABLE = 2, DN_ARRAY = 3, DN_FUNCTION = 4 };

    static std::shared_ptr<DataNode> New(std::string const& uri = "");
    static std::shared_ptr<DataNode> NewEntity(std::string const& uri, std::shared_ptr<DataEntity> const&);
    static std::shared_ptr<DataNode> NewTable(std::string const& uri = "");
    static std::shared_ptr<DataNode> NewArray(std::string const& uri = "");
    static std::shared_ptr<DataNode> NewFunction(std::string const& uri = "");
    /** @addtogroup required @{*/
    virtual eNodeType type() const;
    virtual size_type size() const;
    virtual std::shared_ptr<DataNode> CreateChild() const;
    virtual std::shared_ptr<DataNode> CreateEntity(std::shared_ptr<DataEntity> const&) const;
    virtual std::shared_ptr<DataNode> CreateTable() const;
    virtual std::shared_ptr<DataNode> CreateArray() const;
    virtual std::shared_ptr<DataNode> CreateFunction() const;

    virtual std::shared_ptr<DataEntity> GetEntity() const;

    virtual size_type Set(std::string const& uri, std::shared_ptr<DataNode> const& v);
    virtual size_type Add(std::string const& uri, std::shared_ptr<DataNode> const& v);
    virtual size_type Delete(std::string const& s);
    virtual std::shared_ptr<DataNode> Get(std::string const& uri) const;
    virtual size_type Foreach(std::function<size_type(std::string, std::shared_ptr<DataNode>)> const& f) const;

    virtual size_type Set(size_type s, std::shared_ptr<DataNode> const& v);
    virtual size_type Add(size_type s, std::shared_ptr<DataNode> const& v);
    virtual size_type Delete(size_type s);
    virtual std::shared_ptr<DataNode> Get(size_type s) const;

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

    virtual std::shared_ptr<DataNode> GetRoot() const {
        return GetParent() == nullptr ? const_cast<DataNode*>(this)->shared_from_this() : GetParent();
    }
    virtual std::shared_ptr<DataNode> GetParent() const { return nullptr; }

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
        m_node_ = DataNode::NewEntity("", DataLight::New(u));
        return *this;
    }

    template <typename U>
    KeyValue& operator=(std::initializer_list<U> const& u) {
        m_node_ = DataNode::NewEntity("", DataLight::New(u));
        return *this;
    }

    template <typename U>
    KeyValue& operator=(std::initializer_list<std::initializer_list<U>> const& u) {
        m_node_ = DataNode::NewEntity("", DataLight::New(u));
        return *this;
    }
    template <typename U>
    KeyValue& operator=(std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& u) {
        m_node_ = DataNode::NewEntity("", DataLight::New(u));
        return *this;
    }
};

inline KeyValue operator"" _(const char* c, std::size_t n) { return KeyValue(std::string(c)); }

#define SP_DATA_NODE_HEAD(_CLASS_NAME_)                                                                              \
   protected:                                                                                                        \
    _CLASS_NAME_();                                                                                                  \
                                                                                                                     \
   public:                                                                                                           \
    ~_CLASS_NAME_();                                                                                                 \
    explicit _CLASS_NAME_(_CLASS_NAME_ const& other) = delete;                                                       \
    explicit _CLASS_NAME_(_CLASS_NAME_&& other) = delete;                                                            \
    _CLASS_NAME_& operator=(_CLASS_NAME_ const& other) = delete;                                                     \
    _CLASS_NAME_& operator=(_CLASS_NAME_&& other) = delete;                                                          \
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
    std::shared_ptr<_CLASS_NAME_> Root() const {                                                                     \
        return std::dynamic_pointer_cast<this_type>(const_cast<this_type*>(this)->GetRoot());                        \
    };                                                                                                               \
    std::shared_ptr<_CLASS_NAME_> Parent() const {                                                                   \
        return std::dynamic_pointer_cast<this_type>(const_cast<this_type*>(this)->GetParent());                      \
    };

#define SP_DATA_NODE_FUNCTION                                                                                    \
    eNodeType type() const override;                                                                             \
    size_type size() const override;                                                                             \
                                                                                                                 \
    std::shared_ptr<DataNode> CreateChild() const override;                                                      \
                                                                                                                 \
    std::shared_ptr<DataEntity> GetEntity() const override;                                                      \
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

}  // namespace data
}  // namespace simpla

#endif  // SIMPLA_DATAENTRY_H
