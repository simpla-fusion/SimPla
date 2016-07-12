/**
 * @file MeshEntityRange.h
 * @author salmon
 * @date 2016-05-18.
 */

#ifndef SIMPLA_MESH_MESHENTITY_H
#define SIMPLA_MESH_MESHENTITY_H

#include <type_traits>
#include <cassert>

#include "../parallel/Parallel.h"

#include "MeshCommon.h"
#include "MeshEntityId.h"


namespace simpla { namespace mesh
{

class MeshEntityRange
{
    typedef MeshEntityRange this_type;

    class RangeBase;

    template<typename, bool> struct RangeHolder;

    HAS_MEMBER_FUNCTION(range)

    std::shared_ptr<MeshEntityRange> m_next_;
    std::shared_ptr<RangeBase> m_holder_;

public :

    MeshEntityRange()
        : m_next_(nullptr), m_holder_(nullptr) { }

    //****************************************************************************
    // TBB RangeHolder Concept Begin
    template<typename TOther>
    MeshEntityRange(TOther const &other)
        : m_next_(nullptr),
          m_holder_(std::dynamic_pointer_cast<RangeBase>(
              std::make_shared<RangeHolder<TOther,
                                           has_member_function_range<TOther>::value>>(other)))
    {
    }

    template<typename ...Args>
    MeshEntityRange(this_type &other, parallel::tags::split)
        :
        m_next_(nullptr),
        m_holder_(other.m_holder_->split())
    {
        auto *p0 = &m_next_;
        auto p1 = other.m_next_;
        while (p1 != nullptr)
        {
            (*p0) = std::make_shared<MeshEntityRange>();
            (*p0)->m_holder_ = p1->m_holder_->split();
            (*p0)->m_next_ = nullptr;
            p0 = &((*p0)->m_next_);
            p1 = p1->m_next_;
        }

    }

    MeshEntityRange(this_type const &other)
        :
        m_next_(nullptr),
        m_holder_(other.m_holder_ == nullptr ? nullptr : other.m_holder_->clone())
    {
        auto *p0 = &m_next_;
        auto p1 = other.m_next_;
        while (p1 != nullptr)
        {
            (*p0) = std::make_shared<MeshEntityRange>();
            (*p0)->m_holder_ = p1->m_holder_->clone();
            (*p0)->m_next_ = nullptr;
            p0 = &((*p0)->m_next_);
            p1 = p1->m_next_;
        }
    }

    MeshEntityRange(this_type &&other)
        : m_holder_(other.m_holder_), m_next_(other.m_next_)
    {
        other.m_holder_ = nullptr;
        other.m_next_ = nullptr;
    }

    ~MeshEntityRange() { }

    template<typename T> T &as() { return m_holder_->template as<T>(); }
    template<typename T> T const &as() const { return m_holder_->template as<T>(); }

    int num_of_block() const
    {
        auto p = m_next_;
        int count = 1;
        while (p != nullptr)
        {
            ++count;
            p = p->m_next_;
        }
        return count;
    }

    this_type &operator=(this_type const &other)
    {
        this_type(other).swap(*this);
        return *this;
    }

    this_type &swap(this_type &other)
    {
        std::swap(m_holder_, other.m_holder_);
        std::swap(m_next_, other.m_next_);
        return *this;
    }

    void append(std::shared_ptr<MeshEntityRange> p_next)
    {// TODO remove cycle link
        if (p_next == nullptr || p_next->size() == 0) { return; }

        if (m_holder_ == nullptr)
        {
            m_holder_ = p_next->m_holder_;
            p_next = p_next->m_next_;
        }

        if (p_next != nullptr)
        {
            auto p = &m_next_;
            while (*p != nullptr) { p = &((*p)->m_next_); }
            (*p) = p_next;
        }
    }

    template<typename ...Args>
    void append(Args &&... args)
    {
        append(std::make_shared<MeshEntityRange>(std::forward<Args>(args)...));
    }

public:
    static const bool is_splittable_in_proportion = true;

    bool is_divisible() const { return m_holder_->is_divisible(); }

    bool empty() const { return m_holder_ == nullptr || m_holder_->empty(); }

    size_t size() const
    {
        size_t res = m_holder_ == nullptr ? 0 : m_holder_->size();
        auto p = m_next_;
        while (p != nullptr)
        {
            res += p->m_holder_->size();
            p = p->m_next_;
        }
        return res;
    }

    template<typename T, typename ...Args,
        typename std::enable_if<!std::is_base_of<RangeBase, T>::value>::type * = nullptr>
    static MeshEntityRange create(Args &&...args)
    {
        MeshEntityRange res;
        res.m_holder_ = std::dynamic_pointer_cast<RangeBase>(
            RangeHolder<T, has_member_function_range<T>::value>::create(std::forward<Args>(args)...));
        return std::move(res);
    }

    template<typename T, typename ...Args,
        typename std::enable_if<std::is_base_of<RangeBase, T>::value>::type * = nullptr>
    static MeshEntityRange create(Args &&...args)
    {
        MeshEntityRange res;
        res.m_holder_ = std::dynamic_pointer_cast<RangeBase>(std::make_shared<T>(std::forward<Args>(args)...));
        return std::move(res);
    }

    typedef std::function<void(MeshEntityId const &)> foreach_body_type;

    void foreach(foreach_body_type const &body) const
    {

        if (!empty())
        {
            m_holder_->foreach(body);
            auto p = m_next_;
            while (p != nullptr)
            {
                p->m_holder_->foreach(body);
                p = p->m_next_;
            }
        }
    }


private:


    struct RangeBase
    {
        virtual std::shared_ptr<RangeBase> clone() const = 0;

        virtual std::shared_ptr<RangeBase> split() = 0;

        virtual bool is_a(std::type_info const &) const = 0;

        virtual bool is_divisible() const = 0;

        virtual size_t size() const = 0;

        virtual bool empty() const = 0;

        template<typename T> T &as()
        {
            assert(is_a(typeid(T)));

            return static_cast<RangeHolder<T, has_member_function_range<T>::value> * >(this)->self();

        }

        template<typename T> T const &as() const
        {
            assert(is_a(typeid(T)));

            return static_cast<RangeHolder<T, has_member_function_range<T>::value> const *>(this)->self();
        }

        virtual void foreach(foreach_body_type const &body) const = 0;
    };

    template<typename TOtherRange>
    struct RangeHolder<TOtherRange, false>: public RangeBase
    {
        typedef RangeHolder<TOtherRange, false> this_type;
        TOtherRange m_range_;
    public:

        RangeHolder(TOtherRange const &other)
            : m_range_(other) { }

        template<typename ...Args>
        RangeHolder(this_type &other, Args &&...args)
            : m_range_(other.m_range_, std::forward<Args>(args)...) { }

        virtual  ~RangeHolder() { }

        TOtherRange &self() { return m_range_; }

        TOtherRange const &self() const { return m_range_; }

        template<typename ...Args>
        static std::shared_ptr<this_type> create(Args &&...args)
        {
            return std::shared_ptr<this_type>(new this_type{TOtherRange(std::forward<Args>(args)...)});
        }

        virtual std::shared_ptr<RangeBase> clone() const
        {
            return std::dynamic_pointer_cast<RangeBase>(std::make_shared<this_type>(*this));
        }

        virtual std::shared_ptr<RangeBase> split()
        {
            return std::dynamic_pointer_cast<RangeBase>(std::make_shared<this_type>(*this, parallel::tags::split()));
        };

        virtual size_t size() const { return m_range_.size(); }

        virtual bool is_a(std::type_info const &t_info) const final { return typeid(TOtherRange) == t_info; }

        virtual bool is_divisible() const { return m_range_.is_divisible(); }

        virtual bool empty() const { return m_range_.empty(); }

    public:

        virtual void foreach(foreach_body_type const &body) const
        {
            parallel::foreach(m_range_, body);
        };
    };

    template<typename TContainer>
    struct RangeHolder<TContainer, true>: public RangeBase
    {
        typedef typename TContainer::const_range_type range_type;
        typedef RangeHolder<TContainer, true> this_type;
        TContainer *m_container_;
        range_type m_range_;
    public:

        RangeHolder(TContainer const &other)
            : m_container_(new TContainer(other)), m_range_(m_container_->range()) { }

        template<typename ...Args>
        RangeHolder(this_type &other, Args &&...args)
            : m_container_(other.m_container_), m_range_(other.m_range_, std::forward<Args>(args)...) { }

        virtual  ~RangeHolder() { }

        template<typename ...Args> static std::shared_ptr<this_type>
        create(Args &&...  args)
        {
            return std::shared_ptr<this_type>(new this_type{TContainer(std::forward<Args>(args)...)});
        }

        TContainer &self() { return *m_container_; }

        TContainer const &self() const { return *m_container_; }

        virtual std::shared_ptr<RangeBase> clone() const
        {
            return std::dynamic_pointer_cast<RangeBase>(std::make_shared<this_type>(*this));
        }

        virtual std::shared_ptr<RangeBase> split()
        {
            return std::dynamic_pointer_cast<RangeBase>(std::make_shared<this_type>(*this, parallel::tags::split()));
        };

        virtual size_t size() const { return m_container_->size(); }

        virtual bool is_a(std::type_info const &t_info)
        const final { return typeid(TContainer) == t_info; }

        virtual bool is_divisible() const { return m_range_.is_divisible(); }

        virtual bool empty() const { return m_range_.empty(); }

        virtual void foreach(foreach_body_type const &body) const
        {
            parallel::foreach(m_range_, body);
        };
    };
};

}} //namespace simpla { namespace get_mesh

#endif //SIMPLA_MESH_MESHENTITY_H
