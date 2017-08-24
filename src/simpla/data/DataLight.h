//
// Created by salmon on 17-8-20.
//

#ifndef SIMPLA_DATALIGHT_H
#define SIMPLA_DATALIGHT_H

#include <string.h>

#include "DataEntity.h"
#include "DataTraits.h"
#include "simpla/algebra/nTuple.h"
#include "simpla/utilities/FancyStream.h"

namespace simpla {
namespace data {

struct DataLight : public DataEntity {
    SP_DEFINE_FANCY_TYPE_NAME(DataLight, DataEntity);

   protected:
    DataLight() = default;

   public:
    ~DataLight() override = default;
    static std::shared_ptr<DataLight> New();

    template <typename... Args>
    static std::shared_ptr<DataLight> New(Args&&...);

    template <typename U>
    U as() const;
    template <typename U>
    bool isEqualTo(U const& u) const;

    std::type_info const& value_type_info() const override { return typeid(void); };
    size_type value_alignof() const override { return 0; };
    size_type rank() const override { return 0; }
    size_type extents(size_type* d) const override { return rank(); }
    size_type size() const override { return 0; }

    size_type CopyIn(void const* other) override { return 0; }
    size_type CopyOut(void* other) const override { return 0; }
};
template <typename V, typename Enable = void>
class DataLightT {};

template <typename V>
class DataLightT<V> : public DataLight {
    SP_DEFINE_FANCY_TYPE_NAME(DataLightT, DataLight);
    typedef V value_type;
    value_type m_data_;

   protected:
    DataLightT() = default;

    template <typename... Args>
    explicit DataLightT(Args&&... args) : m_data_{std::forward<Args>(args)...} {}

   public:
    ~DataLightT() override = default;

    template <typename... Args>
    static std::shared_ptr<this_type> New(Args&&... args) {
        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...));
    }

    bool equal(DataEntity const& other) const override {
        auto* p = dynamic_cast<DataLightT<value_type> const*>(&other);
        return p != nullptr && (p->m_data_ == m_data_);
    }

    std::ostream& Print(std::ostream& os, int indent) const override { return FancyPrint(os, m_data_, indent); }

    value_type const& value() const { return m_data_; };
    value_type& value() { return m_data_; };

    void* GetPointer() override { return reinterpret_cast<void*>(&m_data_); }
    void const* GetPointer() const override { return reinterpret_cast<void const*>(&m_data_); }

    std::type_info const& value_type_info() const override { return typeid(value_type); };
    size_type value_alignof() const override { return alignof(value_type); };
    size_type value_sizeof() const override { return sizeof(value_type); };

    size_type rank() const override { return 0; }
    size_type extents(size_type* d) const override { return 0; }
    size_type size() const override { return 1; }

    size_type GetAlignOf() const override { return alignof(value_type); }
    size_type CopyIn(void const* other) override {
        m_data_ = *reinterpret_cast<value_type const*>(other);
        return sizeof(value_type);
    }
    size_type CopyOut(void* other) const override {
        *reinterpret_cast<value_type*>(other) = m_data_;
        return sizeof(value_type);
    }

    size_type CopyIn(value_type const& src) {
        m_data_ = (src);
        return sizeof(value_type);
    }
    size_type CopyOut(value_type& other) const {
        other = (m_data_);
        return sizeof(value_type);
    }

    bool isEqualTo(value_type const& other) const { return m_data_ == other; }
};

template <typename V>
class DataLightT<V*> : public DataLight {
    SP_DEFINE_FANCY_TYPE_NAME(DataLightT, DataLight);
    typedef V value_type;
    std::shared_ptr<value_type> m_data_ = nullptr;
    std::vector<size_type> m_extents_;

   protected:
    DataLightT() = default;
    template <typename TI, typename TPtr>
    DataLightT(int ndims, TI const* extents, TPtr d) : m_extents_(extents, extents + ndims), m_data_(d) {
        if (m_data_ == nullptr) { m_data_.reset(new value_type[size()]); }
    }
    template <typename TI>
    DataLightT(int ndims, TI const* extents) : DataLightT(ndims, extents, nullptr) {}

   public:
    ~DataLightT() override = default;
    template <typename... Args>
    static std::shared_ptr<this_type> New(Args&&... args) {
        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...));
    }
    template <typename TI, typename TPtr>
    static std::shared_ptr<this_type> New(int ndims, TI const* extents, TPtr d) {
        return std::shared_ptr<this_type>(new this_type(ndims, extents, d));
    }

    std::ostream& Print(std::ostream& os, int indent) const override {
        return FancyPrint(os, m_data_.get(), m_extents_.size(), &m_extents_[0], indent);
    }

    auto value() const { return m_data_; };
    auto value() { return m_data_; };

    std::type_info const& value_type_info() const override { return typeid(value_type); };
    size_type value_sizeof() const override { return sizeof(value_type); };
    size_type value_alignof() const override { return alignof(value_type); };

    size_type rank() const override { return m_extents_.size(); }

    size_type extents(size_type* d) const override {
        for (int i = 0; i < m_extents_.size(); ++i) { d[i] = m_extents_[i]; }
        return m_extents_.size();
    }
    size_type size() const override {
        size_type s = 1;
        for (auto v : m_extents_) { s *= v; }
        return s;
    }

    bool isContinue() const override { return true; }

    size_type GetAlignOf() const override { return size() * value_alignof(); }

   private:
    template <typename U>
    static size_type _CopyOut(U& dst, value_type const* src, ENABLE_IF(std::rank<U>::value == 0)) {
        return 0;
    };
    template <typename U>
    static size_type _CopyOut(U& dst, value_type const* src, ENABLE_IF(std::rank<U>::value == 1)) {
        for (int i = 0; i < std::extent<U, 0>::value; ++i) { dst[i] = src[i]; }
        return std::extent<U, 0>::value;
    };
    template <typename U>
    static size_type _CopyOut(U& dst, value_type const* src, ENABLE_IF((std::rank<U>::value > 1))) {
        for (int i = 0; i < std::extent<U, 0>::value; ++i) { src += _CopyOut(dst[i], src); }
        return std::extent<U, 0>::value;
    };
    template <typename U>
    static size_type _CopyIn(value_type* dst, U const& src, ENABLE_IF(std::rank<U>::value == 0)) {
        return 0;
    };
    template <typename U>
    static size_type _CopyIn(value_type* dst, U const& src, ENABLE_IF(std::rank<U>::value == 1)) {
        for (int i = 0; i < std::extent<U, 0>::value; ++i) { dst[i] = src[i]; }
        return std::extent<U, 0>::value;
    };
    template <typename U>
    static size_type _CopyIn(value_type* dst, U const& src, ENABLE_IF((std::rank<U>::value > 1))) {
        for (int i = 0; i < std::extent<U, 0>::value; ++i) { dst += _CopyIn(dst, src[i]); }
        return std::extent<U, 0>::value;
    };

    template <typename U>
    static bool _isEqualTo(U const& left, value_type* right, ENABLE_IF(std::rank<U>::value == 0)) {
        return true;
    };
    template <typename U>
    static bool _isEqualTo(U const& left, value_type* right, ENABLE_IF(std::rank<U>::value == 1)) {
        bool res = true;
        for (int i = 0; i < std::extent<U, 0>::value; ++i) { res = res && (left[i] == right[i]); }
        return res;
    };
    template <typename U>
    static bool _isEqualTo(U const& left, value_type* right, ENABLE_IF((std::rank<U>::value > 1))) {
        bool res = true;
        for (int i = 0; res && i < std::extent<U, 0>::value; ++i) {
            res = res && _CopyIn(left, right[i]);
            right += std::extent<U, 0>::value;
        }
        return res;
    };

   public:
    size_type CopyOut(void* dst) const override {
        size_type s = 0;
        if (m_data_ != nullptr && dst != nullptr) {
            s = size();
            memcpy(dst, m_data_.get(), s);
        }
        return s;
    }
    size_type CopyIn(void const* src) override {
        size_type s = 0;
        if (src != nullptr) {
            s = size();
            memcpy(m_data_.get(), src, s);
        }
        return s;
    }

    template <typename U>
    bool isEqualTo(U const& v) const {
        return _isEqualTo(v, m_data_.get());
    }

    template <typename U>
    size_type CopyOut(U& dst) const {
        return m_data_ == nullptr ? 0 : _CopyOut(dst, m_data_.get());
    }
    template <typename U>
    size_type CopyIn(const U& src) {
        return _CopyIn(m_data_.get(), src);
    }

    void* GetPointer() override { return m_data_.get(); }
    void const* GetPointer() const override { return m_data_.get(); }
};
template <>
class DataLightT<std::string*> : public DataLight {
    SP_DEFINE_FANCY_TYPE_NAME(DataLightT, DataLight);
    typedef std::string value_type;
    std::vector<value_type> m_data_;

   protected:
    DataLightT() = default;
    DataLightT(std::initializer_list<char const*> const& d) {
        for (auto const& v : d) { m_data_.push_back(v); }
    }

   public:
    ~DataLightT() override = default;
    template <typename... Args>
    static std::shared_ptr<this_type> New(Args&&... args) {
        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...));
    }

    std::ostream& Print(std::ostream& os, int indent) const override { return FancyPrint(os, m_data_, indent); }

    auto value() const { return m_data_; };
    auto value() { return m_data_; };

    std::type_info const& value_type_info() const override { return typeid(value_type); };
    size_type value_sizeof() const override { return sizeof(value_type); };
    size_type value_alignof() const override { return alignof(value_type); };

    size_type rank() const override { return m_data_.size(); }

    size_type extents(size_type* d) const override {
        if (d == nullptr) { d[0] = m_data_.size(); }
        return 1;
    }
    size_type size() const override { return m_data_.size(); }

    bool isContinue() const override { return false; }

    size_type GetAlignOf() const override { return 0; }

   private:
    template <typename U>
    static size_type _CopyOut(U& dst, value_type const* src, ENABLE_IF(std::rank<U>::value == 0)) {
        return 0;
    };
    template <typename U>
    static size_type _CopyOut(U& dst, value_type const* src, ENABLE_IF(std::rank<U>::value == 1)) {
        for (int i = 0; i < std::extent<U, 0>::value; ++i) { dst[i] = src[i]; }
        return std::extent<U, 0>::value;
    };
    template <typename U>
    static size_type _CopyOut(U& dst, value_type const* src, ENABLE_IF((std::rank<U>::value > 1))) {
        for (int i = 0; i < std::extent<U, 0>::value; ++i) { src += _CopyOut(dst[i], src); }
        return std::extent<U, 0>::value;
    };
    template <typename U>
    static size_type _CopyIn(value_type* dst, U const& src, ENABLE_IF(std::rank<U>::value == 0)) {
        return 0;
    };
    template <typename U>
    static size_type _CopyIn(value_type* dst, U const& src, ENABLE_IF(std::rank<U>::value == 1)) {
        for (int i = 0; i < std::extent<U, 0>::value; ++i) { dst[i] = src[i]; }
        return std::extent<U, 0>::value;
    };
    template <typename U>
    static size_type _CopyIn(value_type* dst, U const& src, ENABLE_IF((std::rank<U>::value > 1))) {
        for (int i = 0; i < std::extent<U, 0>::value; ++i) { dst += _CopyIn(dst, src[i]); }
        return std::extent<U, 0>::value;
    };

    template <typename U>
    static bool _isEqualTo(U const& left, value_type* right, ENABLE_IF(std::rank<U>::value == 0)) {
        return true;
    };
    template <typename U>
    static bool _isEqualTo(U const& left, value_type* right, ENABLE_IF(std::rank<U>::value == 1)) {
        bool res = true;
        for (int i = 0; i < std::extent<U, 0>::value; ++i) { res = res && (left[i] == right[i]); }
        return res;
    };
    template <typename U>
    static bool _isEqualTo(U const& left, value_type* right, ENABLE_IF((std::rank<U>::value > 1))) {
        bool res = true;
        for (int i = 0; res && i < std::extent<U, 0>::value; ++i) {
            res = res && _CopyIn(left, right[i]);
            right += std::extent<U, 0>::value;
        }
        return res;
    };

   public:
    size_type CopyOut(void* dst) const override {
        size_type s = 0;
        return 0;
    }
    size_type CopyIn(void const* src) override { return 0; }

    template <typename U>
    bool isEqualTo(U const& v) const {
        return _isEqualTo(v, &m_data_[0]);
    }

    template <typename U>
    size_type CopyOut(U& dst) const {
        return m_data_.empty() ? 0 : _CopyOut(dst, &m_data_[0]);
    }
    template <typename U>
    size_type CopyIn(const U& src) {
        if (m_data_.empty()) { m_data_.resize(std::extent<U, 0>::value); }
        return _CopyIn(&m_data_[0], src);
    }

    void* GetPointer() override { return &m_data_[0]; }
    void const* GetPointer() const override { return &m_data_[0]; }
};

inline std::shared_ptr<DataLight> make_light(std::string const& u) { return DataLightT<std::string>::New(u); };
inline std::shared_ptr<DataLight> make_light(char const* u) { return DataLightT<std::string>::New(std::string(u)); };

template <typename U>
std::shared_ptr<DataLight> make_light(U const& u,
                                      ENABLE_IF((std::rank<U>::value == 0 && traits::is_light_data<U>::value))) {
    return DataLightT<U>::New(u);
};
template <typename U>
std::shared_ptr<DataLight> make_light(U const& u, ENABLE_IF((std::rank<U>::value > 0 &&
                                                             traits::is_light_data<traits::value_type_t<U>>::value))) {
    return DataLightT<traits::value_type_t<U>*>::New(u);
};
namespace detail {
template <typename U>
void var_copy(U* dst){};
template <typename U, typename First, typename... Others>
void var_copy(U* dst, First const& first, Others&&... others) {
    dst[0] = first;
    var_copy(dst + 1, std::forward<Others>(others)...);
};
}
template <typename U, typename... Others>
std::shared_ptr<DataLight> make_light(U const& first, U const& second, Others&&... others) {
    size_type s = 2 + sizeof...(others);
    auto d = std::shared_ptr<U>(new U[s]);
    d.get()[0] = first;
    d.get()[second] = first;
    detail::var_copy(d.get() + 2, std::forward<Others>(others)...);
    return DataLightT<U*>::New(1, &s, d);
};
inline std::shared_ptr<DataLight> make_light(std::initializer_list<char const*> const& u) {
    size_type s = u.size();

    return DataLightT<std::string*>::New(u);
};
template <typename U>
std::shared_ptr<DataLight> make_light(std::initializer_list<U> const& u) {
    size_type s = u.size();
    auto res = DataLightT<U*>::New(1, &s);
    size_type i = 0;
    for (auto const& v : u) {
        res->value().get()[i] = v;
        ++i;
    }
    return res;
};
template <typename U>
std::shared_ptr<DataLight> make_light(std::initializer_list<std::initializer_list<U>> const& u) {
    size_type extents[2] = {1, 1};
    extents[0] = u.size();
    for (auto const& v : u) {
        if (extents[1] < v.size()) { extents[1] = v.size(); }
    }
    auto res = DataLightT<U*>::New(2, extents);
    auto* p = res->value().get();
    auto snan = std::numeric_limits<U>::signaling_NaN();
    for (auto const& u1 : u) {
        size_type s = 0;
        for (auto const& v : u1) {
            *p = v;
            ++s;
            ++p;
        }
        for (; s < extents[1]; ++s) {
            *p = snan;
            ++p;
        }
    }
    return res;
};
template <typename U>
std::shared_ptr<DataLight> make_light(std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& u) {
    size_type extents[3] = {1, 1, 1};
    extents[0] = u.size();
    auto res = DataLightT<U*>::New(3, extents);
    return res;
};

template <typename... Args>
std::shared_ptr<DataLight> DataLight::New(Args&&... args) {
    return make_light(std::forward<Args>(args)...);
}
template <typename U>
U DataLight::as() const {
    U res;
    typedef std::conditional_t<std::rank<U>::value == 0, DataLightT<U>, DataLightT<traits::value_type_t<U>*>> type;
    auto const* p = dynamic_cast<type const*>(this);
    if (p == nullptr) { BAD_CAST << typeid(type).name() << std::endl; }
    p->CopyOut(res);
    return res;
}
template <typename U>
bool DataLight::isEqualTo(U const& u) const {
    typedef std::conditional_t<std::rank<U>::value == 0, DataLightT<U>, DataLightT<U*>> type;
    auto const* p = dynamic_cast<type const*>(this);
    return (p != nullptr) && p->isEqualTo(u);
}

//
// template <typename V, int N>
// class DataLightT<nTuple<V, N>> : public DataLight {
//    SP_DEFINE_FANCY_TYPE_NAME(DataLightT, DataLight);
//    typedef V value_type;
//    typedef nTuple<V, N> data_type;
//    data_type m_data_;
//
//   protected:
//    DataLightT() = default;
//
//    template <typename... Args>
//    explicit DataLightT(Args&&... args) : m_data_{std::forward<Args>(args)...} {}
//    //    template <typename U>
//    //    explicit DataLightT(std::initializer_list<U> const& u) : m_data_{u} {}
//
//   public:
//    ~DataLightT() override = default;
//    //    SP_DEFAULT_CONSTRUCT(DataLightT);
//    template <typename... Args>
//    static std::shared_ptr<this_type> New(Args&&... args) {
//        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...));
//    }
//
//    bool equal(DataEntity const& other) const override {
//        auto* p = dynamic_cast<DataLightT<value_type> const*>(&other);
//        return p != nullptr && (p->m_data_ == m_data_);
//    }
//    virtual bool value_equal(void const* other, std::type_info const& info) const override {
//        return value_type_info() == info && m_data_ == *reinterpret_cast<value_type const*>(other);
//    }
//    std::ostream& Print(std::ostream& os, int indent) const override { return FancyPrint(os, m_data_, indent); }
//
//    data_type const& value() const { return m_data_; };
//    data_type& value() { return m_data_; };
//    void* GetPointer() override { return reinterpret_cast<void*>(&m_data_[0]); }
//    void const* GetPointer() const override { return reinterpret_cast<void const*>(&m_data_[0]); }
//    std::type_info const& value_type_info() const override { return typeid(value_type); };
//    size_type value_alignof() const override { return sizeof(value_type); };
//    size_type rank() const override { return 1; }
//    size_type extents(size_type* d) const override {
//        if (d != nullptr) { d[0] = static_cast<size_type>(N); };
//        return rank();
//    }
//    size_type size() const override { return static_cast<size_type>(N); }
//
//    std::experimental::any any() const override { return std::experimental::any(m_data_); };
//};

}  // namespace data
}  // namespace simpla
#endif  // SIMPLA_DATALIGHT_H
