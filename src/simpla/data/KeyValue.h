//
// Created by salmon on 17-1-16.
//

#ifndef SIMPLA_KEYVALUE_H
#define SIMPLA_KEYVALUE_H

#include "DataEntity.h"
namespace simpla {
namespace data {
class KeyValue;
namespace traits {
template <>
struct data_cast<std::initializer_list<KeyValue>> {
    static std::shared_ptr<DataEntity> create(std::initializer_list<KeyValue> const& c);
};
}  // namespace traits{
class KeyValue {
   private:
    std::string m_key_;
    std::shared_ptr<DataEntity> m_value_;

   public:
    KeyValue(char const* k) : m_key_(k), m_value_(traits::data_cast<bool>::create(true)) {}
    KeyValue(std::string const& k) : m_key_(k), m_value_(traits::data_cast<bool>::create(true)) {}
    template <typename U>
    KeyValue(std::string const& k, U const& u) : m_key_(k), m_value_(traits::data_cast<U>::create(u)) {}
    KeyValue(KeyValue const& other) : m_key_(other.m_key_), m_value_(other.m_value_) {}
    KeyValue(KeyValue&& other) : m_key_(other.m_key_), m_value_(other.m_value_) {}
    ~KeyValue() {}

    template <typename U>
    KeyValue& operator=(U const& u) {
        m_value_ = traits::data_cast<U>::create(u);
        return *this;
    }

    KeyValue& operator=(char const* c) {
        m_value_ = traits::data_cast<std::string>::create(c);
        return *this;
    }
    KeyValue& operator=(char* c) {
        m_value_ = traits::data_cast<std::string>::create(c);
        return *this;
    }
    KeyValue& operator=(std::initializer_list<KeyValue> const& u) {
        m_value_ = traits::data_cast<std::initializer_list<KeyValue>>::create(u);
        return *this;
    }

    std::string const& key() const { return m_key_; }
    std::shared_ptr<DataEntity> const& value() const { return m_value_; }
};

inline KeyValue operator"" _(const char* c, std::size_t n) {
    return KeyValue{std::string(c), traits::data_cast<std::string>::create(c)};
}

}  // namespace data
}  // namespace simpla

#endif  // SIMPLA_KEYVALUE_H
