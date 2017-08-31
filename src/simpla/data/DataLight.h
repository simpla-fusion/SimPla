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
    static std::shared_ptr<DataLight> New() { return std::shared_ptr<DataLight>(new DataLight); }

    template <typename... Args>
    static std::shared_ptr<DataLight> New(Args&&...);

    std::type_info const& value_type_info() const override { return typeid(void); };
    size_type value_alignof() const override { return 0; };
    size_type rank() const override { return 0; }
    size_type extents(size_type* d) const override { return rank(); }
    size_type size() const override { return 0; }
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
    auto const* pointer() const { return &m_data_; };
    auto* pointer() { return &m_data_; };
    void* GetPointer() override { return reinterpret_cast<void*>(&m_data_); }
    void const* GetPointer() const override { return reinterpret_cast<void const*>(&m_data_); }

    std::type_info const& value_type_info() const override { return typeid(value_type); };
    size_type value_alignof() const override { return alignof(value_type); };
    size_type value_sizeof() const override { return sizeof(value_type); };

    size_type rank() const override { return 0; }
    size_type extents(size_type* d) const override { return 0; }
    size_type size() const override { return 1; }

    size_type GetAlignOf() const override { return alignof(value_type); }

    size_type CopyIn(value_type const& src) {
        m_data_ = src;
        return 1;
    }
    size_type CopyOut(value_type& other) const {
        other = m_data_;
        return 1;
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

    auto const& value() const { return m_data_; };
    auto& value() { return m_data_; };
    auto const* pointer() const { return m_data_.get(); };
    auto* pointer() { return m_data_.get(); };

    void* GetPointer() override { return m_data_.get(); }
    void const* GetPointer() const override { return m_data_.get(); }

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
        dst = *src;
        return 1;
    };

    template <typename U>
    static size_type _CopyOut(U& dst, value_type const* src, ENABLE_IF((std::rank<U>::value > 0))) {
        size_type count = 0;
        for (int i = 0; i < std::extent<U, 0>::value; ++i) { count += _CopyOut(dst[i], src + count); }
        return count;
    };
    template <typename U>
    static size_type _CopyIn(value_type* dst, U const& src, ENABLE_IF(std::rank<U>::value == 0)) {
        *dst = src;
        return 1;
    };
    template <typename U>
    static size_type _CopyIn(value_type* dst, U const& src, ENABLE_IF((std::rank<U>::value > 0))) {
        size_type count = 0;
        for (int i = 0; i < std::extent<U, 0>::value; ++i) { count += _CopyIn(dst + count, src[i]); }
        return count;
    };

    template <typename U>
    static size_type _isEqualTo(U const& left, value_type* right, bool* res, ENABLE_IF(std::rank<U>::value == 0)) {
        *res = *res && (left == *right);
        return 1;
    };

    template <typename U>
    static size_type _isEqualTo(U const& left, value_type* right, bool* res, ENABLE_IF((std::rank<U>::value > 0))) {
        size_type count = 0;
        for (int i = 0; i < std::extent<U, 0>::value; ++i) { count += _isEqualTo(left, right[i], res); }
        return count;
    };

   public:
    template <typename U>
    bool isEqualTo(U const& v) const {
        bool res = true;
        _isEqualTo(v, m_data_.get(), &res);
        return res;
    }

    template <typename U>
    size_type CopyOut(U& dst) const {
        return m_data_ == nullptr ? 0 : _CopyOut(dst, m_data_.get());
    }
    template <typename U>
    size_type CopyIn(const U& src) {
        return _CopyIn(m_data_.get(), src);
    }
};
template <>
class DataLightT<std::string*> : public DataLight {
    SP_DEFINE_FANCY_TYPE_NAME(DataLightT, DataLight);
    typedef std::string value_type;
    std::vector<value_type> m_data_;
    std::vector<size_type> m_extents_;

   protected:
    DataLightT() = default;
    DataLightT(std::initializer_list<char const*> const& d) {
        for (auto const& v : d) { m_data_.push_back(std::string(v)); }
    }
    DataLightT(int rank, size_type const* extents) {}

   public:
    ~DataLightT() override = default;
    template <typename... Args>
    static std::shared_ptr<this_type> New(Args&&... args) {
        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...));
    }

    std::ostream& Print(std::ostream& os, int indent) const override { return FancyPrint(os, m_data_, indent); }

    auto const& value() const { return m_data_; };
    auto& value() { return m_data_; };
    value_type const* pointer() const { return &m_data_[0]; };
    value_type* pointer() { return &m_data_[0]; };
    void* GetPointer() override { return &m_data_[0]; }
    void const* GetPointer() const override { return &m_data_[0]; }

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
};
namespace detail {
inline std::shared_ptr<DataLight> make_light(std::string const& u) { return DataLightT<std::string>::New(u); };
inline std::shared_ptr<DataLight> make_light(char const* u) { return DataLightT<std::string>::New(std::string(u)); };

template <typename U>
std::shared_ptr<DataLight> make_light(U const& u,
                                      ENABLE_IF((std::rank<U>::value == 0 && traits::is_light_data<U>::value))) {
    return DataLightT<U>::New(u);
};
template <typename U>
void get_extents(U const& v, size_type* extents, ENABLE_IF((std::rank<U>::value == 0))) {}
template <typename U>
void get_extents(U const& v, size_type* extents, ENABLE_IF((std::rank<U>::value > 0))) {
    extents[0] = std::extent<U, 0>::value;
    get_extents(v[0], extents + 1);
}

template <typename U>
std::shared_ptr<DataLight> make_light(U const& u, ENABLE_IF((std::rank<U>::value > 0 &&
                                                             traits::is_light_data<traits::value_type_t<U>>::value))) {
    auto* extents = new size_type[std::rank<U>::value];
    get_extents(u, extents);
    auto res = DataLightT<traits::value_type_t<U>*>::New(std::rank<U>::value, extents);
    res->CopyIn(u);
    return res;
};

inline std::shared_ptr<DataLight> make_light(std::initializer_list<char const*> const& u) {
    size_type s = u.size();
    auto res = DataLightT<std::string*>::New(1, &s);
    for (auto const& v : u) { res->value().push_back(std::string(v)); }
    return res;
};

template <typename V, typename U>
size_type CopyND(V* dst, U const& src, size_type* extents) {
    if (dst != nullptr) { *dst = src; }
    return 1;
}
template <typename V, typename U>
size_type CopyND(V* dst, std::initializer_list<U> const& src, size_type* extents) {
    static auto snan = std::numeric_limits<traits::value_type_t<U>>::signaling_NaN();

    size_type count = 0;
    if (dst == nullptr) { extents[0] = std::max(extents[0], src.size()); }

    for (auto const& v : src) { count += CopyND(dst == nullptr ? nullptr : dst + count, v, extents + 1); }

    if (dst != nullptr) {
        for (size_type s = src.size(); s < extents[0]; ++s) { count += CopyND(dst + count, snan, extents + 1); }
    }
    return count;
}

template <typename U>
std::shared_ptr<DataLight> make_light(std::initializer_list<U> const& u) {
    size_type extents = u.size();
    auto res = DataLightT<traits::value_type_t<U>*>::New(1, &extents);
    CopyND(res->pointer(), u, &extents);
    return res;
};
template <typename U>
std::shared_ptr<DataLight> make_light(std::initializer_list<std::initializer_list<U>> const& u) {
    size_type rank = 2;
    size_type extents[2] = {1, 1};
    CopyND(static_cast<U*>(nullptr), u, extents);
    auto res = DataLightT<traits::value_type_t<U>*>::New(rank, extents);
    CopyND(res->pointer(), u, extents);
    return res;
};
template <typename U>
std::shared_ptr<DataLight> make_light(std::initializer_list<std::initializer_list<std::initializer_list<U>>> const& u) {
    size_type rank = 3;
    size_type extents[3] = {1, 1, 1};
    detail::CopyND(static_cast<U*>(nullptr), u, extents);
    auto res = DataLightT<traits::value_type_t<U>*>::New(rank, extents);
    detail::CopyND(res->pointer(), u, extents);
    return res;
};
}  // namespace detail
template <typename... Args>
std::shared_ptr<DataLight> DataLight::New(Args&&... args) {
    return detail::make_light(std::forward<Args>(args)...);
}
// template <typename U>
// size_type DataLight::as(U* res) const {
//    size_type count = 0;
//    typedef std::conditional_t<std::rank<U>::value == 0, DataLightT<U>, DataLightT<traits::value_type_t<U>*>> type;
//    auto const* p = dynamic_cast<type const*>(this);
//    if (p != nullptr) { count = p->CopyOut(*res); }
//    return count;
//}
// template <typename U>
// bool DataLight::isEqualTo(U const& u) const {
//    typedef std::conditional_t<std::rank<U>::value == 0, DataLightT<U>, DataLightT<U*>> type;
//    auto const* p = dynamic_cast<type const*>(this);
//    return (p != nullptr) && p->isEqualTo(u);
//}

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
