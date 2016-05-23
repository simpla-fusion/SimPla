//
// Created by salmon on 16-5-18.
//

#ifndef SIMPLA_GTL_DATAVIEW_H
#define SIMPLA_GTL_DATAVIEW_H

#include <memory>
#include <cassert>

namespace simpla { namespace gtl
{
namespace detail
{
struct NullHashFunction
{
    template<typename T, typename ...Others>
    inline constexpr T const &operator()(T const &v, Others &&...) const { return v; }
};
} //namespace detail


template<typename T, typename HashFunction=detail::NullHashFunction>
class DataView
{
    typedef DataView<T, HashFunction> this_type;
public:
    typedef T value_type;

    template<typename ...Args>
    DataView(Args &&...args)
            : m_hash_fun_(std::forward<Args>(args)...) { }

    template<typename ...Args>
    DataView(std::shared_ptr<value_type> d, Args &&...args)
            : m_hash_fun_(std::forward<Args>(args)...), m_data_(d) { }

    DataView(this_type const &other)
            : m_hash_fun_(other.m_hash_fun_), m_data_(other.m_data_), m_max_hash_(other.m_max_hash_) { }

    DataView(this_type &&other)
            : m_hash_fun_(other.m_hash_fun_), m_data_(other.m_data_), m_max_hash_(other.m_max_hash_) { }

    ~DataView() { }

    this_type &operator=(this_type &other)
    {
        this_type(other).swap(*this);
        return *this;
    }

    void swap(this_type &other)
    {
        std::swap(m_data_, other.m_data_);
        std::swap(m_hash_fun_, other.m_hash_fun_);
        std::swap(m_max_hash_, other.m_max_hash_);
    }

    size_t size() { return m_max_hash_; }

    HashFunction const &HashFunction() const { return m_hash_fun_; }

    std::shared_ptr<value_type> data() { return m_data_; }

    std::shared_ptr<value_type> data() const { return m_data_; }

    template<typename Key>
    value_type &operator[](Key const &key) { return get(m_hash_fun_(key)); }

    template<typename Key>
    value_type const &operator[](Key const &key) const { return get(m_hash_fun_(key)); }

    template<typename ...Args>
    value_type &operator()(Args &&...args) { return get(m_hash_fun_(std::forward<Args>(args)...)); }

    template<typename ...Args>
    value_type &operator()(Args &&...args) const { return get(m_hash_fun_(std::forward<Args>(args)...)); }

private:
    HashFunction m_hash_fun_;
    std::shared_ptr<value_type> m_data_{nullptr};
    size_t m_max_hash_ = 0;

    inline value_type &get(size_t s)
    {
        assert(m_max_hash_ > 0 && s < m_max_hash_);
        assert(m_max_hash_ > 0 && m_data_ != nullptr);
        return m_data_.get()[s];
    }

    inline value_type const &get(size_t s) const
    {
        assert(m_max_hash_ > 0 && s < m_max_hash_);
        assert(m_max_hash_ > 0 && m_data_ != nullptr);
        return m_data_.get()[s];
    }

};
}}//namespace simpla { namespace gtl

#endif //SIMPLA_GTL_DATAVIEW_H
