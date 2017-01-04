//
// Created by salmon on 17-1-1.
//

#ifndef SIMPLA_RANGE_H
#define SIMPLA_RANGE_H

#include <simpla/concept/Splittable.h>
#include <cstddef>
#include <iterator>
#include <memory>
#include "check_concept.h"

namespace simpla {
namespace range {
/**
 * @brief everthing about '''range''' concept,
 * @note  DO NOT EXPORSE THIS NAMESPACE!!
 *
 * @tparam TRange
 */

template <typename TRange>
struct Iterator_;
template <typename TRange>
struct ConstIterator_;
template <typename TRange>
struct RangeHolder;
template <typename U, typename TRange>
struct RangeProxy;

template <typename _Tp, typename _Category = std::input_iterator_tag,
          typename _Distance = ptrdiff_t, typename _Pointer = _Tp*,
          typename _Reference = _Tp&>
struct Range_ {
   private:
    typedef Range_<_Tp, _Category, _Distance, _Pointer, _Reference> this_type;

   public:
    typedef Iterator_<this_type> iterator;
    typedef ConstIterator_<this_type> const_iterator;

    /// One of the @link iterator_tags tag types@endlink.
    typedef _Category iterator_category;
    /// The type "pointed to" by the iterator.
    typedef _Tp value_type;
    /// Distance between iterators is represented as this type.
    typedef _Distance difference_type;
    /// This type represents a pointer-to-value_type.
    typedef _Pointer pointer;
    /// This type represents a reference-to-value_type.
    typedef _Reference reference;

    Range_() : m_holder_(nullptr) {}

    Range_(Range_& r, concept::tags::split)
        : m_holder_(new Range_(r.Range_())) {}

    Range_(Range_ const& r) : m_holder_(r.m_holder_) {}

    Range_(Range_&& r) : m_holder_(r.m_holder_) {}

    iterator begin() const { return begin(*m_holder_); };

    iterator end() const { return end(*m_holder_); };

    bool is_divisible() { return m_holder_->is_divisible(); }

    virtual size_type size() const { return m_holder_->size(); }

    bool empty() const { return m_holder_->empty(); }

    void swap(this_type& other) { std::swap(m_holder_, other.m_holder_); }

    template <typename U, typename... Args>
    static this_type create(Args&&... args) {
        this_type res;
        res.m_holder_ = std::dynamic_pointer_cast<RangeHolder<this_type>>(
            std::make_shared<RangeProxy<U, this_type>>(
                std::forward<Args>(args)...));
        return std::move(res);
    }

    this_type split(concept::tags::split const& s = concept::tags::split()) {
        this_type res;
        res.m_holder_ = m_holder_->split(s);
        return std::move(res);
    }

    this_type contact(this_type& other) {
        this_type res(*this);
        res.append(other);
        return std::move(res);
    }

    void append(this_type const& other) {
        if (m_holder_ != nullptr) {
            m_holder_->append(other.m_holder_);
        } else {
            m_holder_ = other.m_holder_;
        }
    }

    //    void append(this_type &&other) { m_holder_->append(other.m_holder_); }

    template <typename U>
    U& as() {
        return m_holder_->template as<U>();
    }

    template <typename U>
    U const& as() const {
        return m_holder_->template as<U>();
    }

    std::shared_ptr<RangeHolder<this_type>> next() {
        return m_holder_->m_next_;
    }

    template <typename TFun>
    void foreach (TFun const&) const {}

   private:
    std::shared_ptr<RangeHolder<this_type>> m_holder_;
};

template <typename TRange>
struct Iterator_ {};
template <typename TRange>
struct ConstIterator_ {};

template <typename _TRange>
struct RangeHolder : public std::enable_shared_from_this<RangeHolder<_TRange>> {
    typedef RangeHolder<_TRange> this_type;

   public:
    typedef _TRange range_type;
    typedef typename range_type::iterator iterator;
    typedef typename range_type::const_iterator const_iterator;

    std::shared_ptr<this_type> split(concept::tags::split const& s) {}

    virtual iterator begin() const = 0;

    virtual iterator end() const = 0;

    virtual const_iterator cbegin() const = 0;

    virtual const_iterator cend() const = 0;

    virtual size_type size() const = 0;

    virtual bool is_divisible() = 0;

    virtual bool empty() const = 0;

    virtual void append(std::shared_ptr<this_type> const& p) {
        p->m_next_ = m_next_;
        m_next_ = p;
    }

    virtual void* data() = 0;

    virtual void const* data() const = 0;

    template <typename U>
    U& as() {
        return *reinterpret_cast<U*>(data());
    }

    std::shared_ptr<this_type> next() { return m_next_; }

   private:
    std::shared_ptr<this_type> m_next_ = nullptr;
};

template <typename U, typename TRange>
struct RangeProxy : public RangeHolder<TRange> {
    typedef TRange range_type;
    typedef typename range_type::value_type value_type;
    typedef typename range_type::iterator iterator;
    typedef typename range_type::const_iterator const_iterator;

    template <typename... Args>
    RangeProxy(Args&&... args) : m_self_(new U(std::forward<Args>(args)...)) {}

    template <typename TFun>
    void apply(TFun const& fun) const {
        static_assert(traits::is_callable<TFun(value_type&)>::value,
                      "Function is not  applicable! ");

        //        for (auto it = m_begin_; it != m_end_; ++it) { fun(*it); }
    }

    virtual iterator begin() const {};

    virtual iterator end() const {};

    virtual const_iterator cbegin() const {};

    virtual const_iterator cend() const {};

    virtual bool is_divisible() { return false; };

    virtual bool empty() const { return false; };

    virtual size_type size() const { return 0; }

    virtual void* data() { return reinterpret_cast<void*>(m_self_.get()); };

    virtual void const* data() const {
        return reinterpret_cast<void*>(m_self_.get());
    };

    std::unique_ptr<U> m_self_;
};
}
}  // namespace simpla{namespace range
namespace simpla {
template <typename _Tp, typename _Category = std::input_iterator_tag,
          typename _Distance = ptrdiff_t, typename _Pointer = _Tp*,
          typename _Reference = _Tp&>
using Range = range::Range_<_Tp, _Category, _Distance, _Pointer, _Reference>;
}
#endif  // SIMPLA_RANGE_H
