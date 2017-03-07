//
// Created by salmon on 17-1-1.
//

#ifndef SIMPLA_RANGE_H
#define SIMPLA_RANGE_H

#include <simpla/concept/Splittable.h>
#include <simpla/engine/SPObject.h>
#include <simpla/toolbox/Log.h>
#include <cstddef>
#include <iterator>
#include <memory>
#include "simpla/concept/CheckConcept.h"

namespace simpla {

/**
 * @brief Range concept
 *   @ref https://github.com/ericniebler/range-v3
 *   @ref TBB Range concept https://software.intel.com/en-us/node/506143
 *
 *   - lightweight
 *   - non-owning
 *   - range may/may not be iteraterable (begin,end may not be defined)
 *   - range may/may not be splittable
 *   - range may be iterated in an arbitrary order, or in parallel
 *   example:
 *     double d[10]
 *     range<double> r(d,d+10);

 *     r.foreach([&](double & v){ r+=1.0;});
 *
 *     range<double> r1(r,tags::split());
 *
 *     auto r2=r1.split();
 *
 *     range<const double> cr(d,d+10);
 *     r.foreach([&](double const & v){ r+=1.0;});
 *
 * @tparam _Category
 * @tparam _Tp  value_type
 * @tparam _Distance difference_type
 * @tparam _Pointer  value_type *
 * @tparam _Reference value_type &
 */

template <typename...>
struct ContinueRange;
template <typename...>
struct UnorderedRange;

template <typename T>
struct RangeBase {
    SP_OBJECT_BASE(RangeBase<T>);

   private:
    typedef ContinueRange<T> continue_type;
    typedef UnorderedRange<T> unordered_type;

   public:
    typedef T value_type;
    virtual bool is_divisible() const { return false; }

    virtual size_type size() const { return 0; }

    virtual bool empty() const { return true; }

    virtual void foreach_override(std::function<void(value_type&)> const& fun) const { DO_NOTHING; };

    virtual std::shared_ptr<this_type> split(concept::tags::split const& sp) {
        UNIMPLEMENTED;
        return nullptr;
    }

    template <typename TFun, typename... Args>
    void foreach (TFun const& fun, Args && ... args) const {
        if (isA(typeid(continue_type))) {
            cast_as<continue_type>()->foreach (fun, std::forward<Args>(args)...);
        } else if (isA(typeid(unordered_type))) {
            cast_as<unordered_type>()->foreach (fun, std::forward<Args>(args)...);
        } else {
            foreach_override([&](value_type& v) { fun(v, std::forward<Args>(args)...); });
        }
    }

    std::shared_ptr<RangeBase<T>> m_next_;
};

template <typename T>
struct ContinueRange<T> : public RangeBase<T> {
    SP_OBJECT_HEAD(ContinueRange<T>, RangeBase<T>)
    typedef T value_type;

   public:
    virtual bool is_divisible() const { return false; }

    virtual size_type size() const { return 0; }

    virtual bool empty() const { return true; }

    virtual void foreach_override(std::function<void(value_type&)> const& fun) const { UNIMPLEMENTED; };

    virtual std::shared_ptr<base_type> split(concept::tags::split const& sp) {
        UNIMPLEMENTED;
        return nullptr;
    }

    template <typename TFun, typename... Args>
    void foreach (TFun const& fun, Args && ... args) const {
        UNIMPLEMENTED;
    }
};

template <typename T>
struct UnorderedRange<T> : public RangeBase<T> {
    SP_OBJECT_HEAD(UnorderedRange<T>, RangeBase<T>)
    typedef T value_type;

   public:
    virtual bool is_divisible() const { return false; }

    virtual size_type size() const { return 0; }

    virtual bool empty() const { return true; }

    virtual void foreach_override(std::function<void(value_type&)> const& fun) const { UNIMPLEMENTED; };

    virtual std::shared_ptr<base_type> split(concept::tags::split const& sp) {
        UNIMPLEMENTED;
        return nullptr;
    }

    template <typename TFun, typename... Args>
    void foreach (TFun const& fun, Args && ... args) const {
        UNIMPLEMENTED;
    }
    void insert(value_type const&) {}
};

template <typename TOtherRange>
struct RangeAdapter : public RangeBase<typename TOtherRange::value_type>, public TOtherRange {
    typedef typename TOtherRange::value_type value_type;

    SP_OBJECT_HEAD(RangeAdapter<TOtherRange>, RangeBase<value_type>)

    template <typename... Args>
    RangeAdapter(Args&&... args) : TOtherRange(std::forward<Args>(args)...) {}

    RangeAdapter(TOtherRange& other, concept::tags::split const& sp) : TOtherRange(other.split(sp)) {}

    virtual ~RangeAdapter() {}

    virtual bool is_divisible() { return TOtherRange::is_divisible(); };

    virtual bool empty() const { return TOtherRange::empty(); };

    virtual size_type size() const { return TOtherRange::size(); }

    std::shared_ptr<base_type> split(concept::tags::split const& sp) {
        return std::dynamic_pointer_cast<base_type>(std::make_shared<this_type>(*this, sp));
    }
    virtual void foreach_override(std::function<void(value_type&)> const& fun) const {
        for (auto& item : *this) { fun(item); }
    };
};

template <typename TIterator>
struct IteratorRange : public RangeBase<typename std::iterator_traits<TIterator>::value_type> {
    typedef TIterator iterator;

    typedef typename std::iterator_traits<TIterator>::value_type value_type;

    SP_OBJECT_HEAD(IteratorRange<TIterator>, RangeBase<value_type>)

   public:
    IteratorRange(iterator const& b, iterator const& e) : m_b_(b), m_e_(e) { ASSERT(std::distance(m_b_, m_e_) >= 0); }

    IteratorRange(this_type& other) : m_b_(other.m_b_), m_e_(other.m_e_) {}

    IteratorRange(this_type&& other) : m_b_(other.m_b_), m_e_(other.m_e_) {}

    virtual ~IteratorRange() {}

    bool is_divisible() const { return size() > 0; }

    size_type size() const { return static_cast<size_type>(std::distance(m_e_, m_b_)); }

    bool empty() const { return size() > 0; }

    std::shared_ptr<base_type> split(concept::tags::split const& sp) {
        iterator b = m_b_;
        m_b_ = std::distance(m_e_, m_b_) * sp.left() / (sp.left() + sp.right());
        return std::dynamic_pointer_cast<base_type>(std::make_shared<this_type>(b, m_b_));
    }

    void foreach_override(std::function<void(value_type&)> const& fun) const {
        for (auto it = m_b_; it != m_e_; ++it) { fun(*it); }
    }

   private:
    iterator m_b_, m_e_;
};

template <typename T>
struct Range {
    typedef Range<T> this_type;
    typedef RangeBase<T> base_type;

   public:
    typedef T value_type;

    Range() : m_next_(nullptr) {}
    explicit Range(std::shared_ptr<base_type> const& p) : m_next_(p) {}
    Range(this_type const& other) : m_next_(other.m_next_) {}
    Range(this_type&& other) : m_next_(other.m_next_) {}
    Range(this_type& other, concept::tags::split const& s) : Range(other.split(s)) {}
    ~Range() {}

    void swap(this_type& other) { std::swap(m_next_, other.m_next_); }

    bool is_divisible() const {  // FIXME: this is not  full functional
        return m_next_ != nullptr && m_next_->is_divisible();
    }

    bool empty() const {  // FIXME: this is not  full functional
        return m_next_ == nullptr || m_next_->empty();
    }

    this_type split(concept::tags::split const& s = concept::tags::split()) {
        // FIXME: this is not  full functional
        this_type res;
        UNIMPLEMENTED;
        return std::move(res);
    }

    void append(this_type const& other) { append(other.m_next_); }

    void append(std::shared_ptr<base_type> const& other) {
        auto& cursor = m_next_;
        while (cursor != nullptr) { cursor = cursor->m_next_; }
        cursor = other;
    }
    size_type size() const {
        size_type res = 0;
        for (auto* cursor = &m_next_; *cursor != nullptr; cursor = &((*cursor)->m_next_)) { res += (*cursor)->size(); }
        return res;
    }

    template <typename... Args>
    void foreach (Args&&... args) const {
        for (auto* cursor = &m_next_; *cursor != nullptr; cursor = &((*cursor)->m_next_)) {
            (*cursor)->foreach (std::forward<Args>(args)...);
        }
    }
    RangeBase<T>& self() { return *m_next_; }
    RangeBase<T> const& self() const { return *m_next_; }

   private:
    std::shared_ptr<RangeBase<T>> m_next_;
};

template <typename T, typename... Args>
Range<T> make_continue_range(Args&&... args) {
    return std::move(Range<T>(std::make_shared<ContinueRange<T>>(std::forward<Args>...)));
};

template <typename T, typename... Args>
Range<T> make_unordered_range(Args&&... args) {
    return Range<T>(std::make_shared<UnorderedRange<T>>(std::forward<Args>...));
};

template <typename TIterator>
Range<typename std::iterator_traits<TIterator>::value_type> make_iterator_range(TIterator const& b,
                                                                                TIterator const& e) {
    return Range<typename std::iterator_traits<TIterator>::value_type>(
        std::make_shared<IteratorRange<TIterator>>(b, e));
};

template <typename TOtherRange, typename... Args>
Range<typename TOtherRange::value_type> make_range(Args&&... args) {
    return Range<typename TOtherRange::value_type>(std::make_shared<RangeAdapter<TOtherRange>>(std::forward<Args>...));
};
}
#endif  // SIMPLA_RANGE_H
