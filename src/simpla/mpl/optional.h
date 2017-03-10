//
// Created by salmon on 17-2-28.
//

#ifndef SIMPLA_OPTIONAL_H
#define SIMPLA_OPTIONAL_H

namespace simpla {

/**
 * @brief  waiting for C++17 std::optional
 * @tparam T
 */
template <typename T>
class optional {
    typedef T value_type;

   public:
    optional() {}
    optional(optional const&) {}
    constexpr const T* operator->() const { return m_value_; };
    constexpr T* operator->() { return m_value_; };
    constexpr const T& operator*() const { return *m_value_; };
    constexpr T& operator*() { return *m_value_; };
    constexpr const T&& operator*() const&& { return std::move(*m_value_); };
    constexpr T&& operator*() && { return std::move(*m_value_); };

   private:
    value_type* m_value_ = nullptr;
};
}
#endif  // SIMPLA_OPTIONAL_H
