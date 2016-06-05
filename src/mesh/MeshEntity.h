/**
 * @file MeshEntity.h
 * @author salmon
 * @date 2016-05-18.
 */

#ifndef SIMPLA_MESH_MESHENTITY_H
#define SIMPLA_MESH_MESHENTITY_H

#include <type_traits>
#include <cassert>
#include "Mesh.h"
#include "../parallel/Parallel.h"

namespace simpla
{
namespace mesh
{
typedef unsigned long MeshEntityId;

typedef long MeshEntityIdDiff;
enum MeshEntityType
{
    VERTEX = 0, EDGE = 1, FACE = 2, VOLUME = 3

//    TRIANGLE = (3 << 2) | 2,
//
//    QUADRILATERAL = (4 << 2) | 2,
//
//    // place Holder
//
//    POLYGON = ((-1) << 2) | 2,
//
//    // custom polygon
//
//
//
//    TETRAHEDRON = (6 << 2) | 3,
//    PYRAMID,
//    PRISM,
//    KNIFE,
//
//    HEXAHEDRON = MAX_POLYGON + 12,
//    // place Holder
//            POLYHEDRON = MAX_POLYGON + (1 << 5),
//    // custom POLYHEDRON
//
//    MAX_POLYHEDRON = MAX_POLYGON + (1 << 6)

};


class MeshEntityIterator
        : public std::iterator<typename std::random_access_iterator_tag, MeshEntityId, MeshEntityIdDiff, MeshEntityId *, MeshEntityId>
{
    typedef MeshEntityIterator this_type;

    struct HolderBase;

    template<typename TOther>
    struct Holder;

    std::shared_ptr<HolderBase> m_holder_;

public:
    MeshEntityIterator() : m_holder_(nullptr) { }

    template<typename TOther>
    MeshEntityIterator(TOther const &it) : m_holder_(
            std::dynamic_pointer_cast<HolderBase>(std::make_shared<Holder<TOther> >(it))) { }

    template<typename TI>
    MeshEntityIterator(TI const &it, TI const &ie) : m_holder_(
            std::dynamic_pointer_cast<HolderBase>(
                    std::make_shared<Holder<std::pair<TI, TI>>>(std::make_pair(it, ie)))) { }

    MeshEntityIterator(this_type const &other) : m_holder_(other.m_holder_->clone()) { }

    MeshEntityIterator(this_type &&other) : m_holder_(other.m_holder_) { }

    virtual  ~MeshEntityIterator() { }

    this_type &operator=(this_type const &other)
    {
        this_type(other).swap(*this);
        return *this;
    }

    void swap(this_type &other) { std::swap(m_holder_, other.m_holder_); }


    const reference   operator*() const { return get(); }

    reference  operator*() { return get(); }

    this_type &operator++()
    {
        advance(1);
        return *this;
    }

    this_type &operator--()
    {
        advance(-1);
        return *this;
    }


    this_type  operator++(int)
    {
        this_type res(*this);
        ++(*this);
        return std::move(res);
    }

    this_type operator--(int)
    {
        this_type res(*this);
        --(*this);
        return std::move(res);
    }

    reference operator[](difference_type n) const
    {
        this_type res(*this);
        res += n;
        return *res;
    }


    this_type &operator+=(difference_type const &n)
    {
        advance(n);
        return *this;
    }

    this_type &operator-=(difference_type const &n)
    {
        advance(-n);
        return *this;
    }


    this_type operator+(difference_type const &n) const
    {
        this_type res(*this);
        res += n;
        return std::move(res);
    }

    this_type operator-(difference_type const &n) const
    {
        this_type res(*this);
        res -= n;
        return std::move(res);
    }


    difference_type operator-(this_type const &other) const { return distance(other); }

    bool operator==(this_type const &other) const { return equal(other); }

    bool operator!=(this_type const &other) const { return !equal(other); }

    bool operator<(this_type const &other) const { return distance(other) < 0; }

    bool operator>(this_type const &other) const { return distance(other) > 0; }

    bool operator<=(this_type const &other) const { return distance(other) <= 0; }

    bool operator>=(this_type const &other) const { return distance(other) >= 0; }


    bool equal(this_type const &other) const
    {
        return (m_holder_ == nullptr && other.m_holder_ == nullptr) || m_holder_->equal(*other.m_holder_);
    }

    /** return  current value */
    const reference get() const { return m_holder_->get(); }

    reference get() { return m_holder_->get(); }

    /** advance iterator n steps, return number of actually advanced steps  */
    void advance(difference_type n = 1) { m_holder_->advance(n); }

    difference_type distance(this_type const &other) const
    {
        return m_holder_->distance(*other.m_holder_);
    };

private:

    struct HolderBase
    {

        virtual std::shared_ptr<HolderBase> clone() const = 0;

        virtual bool is_a(std::type_info const &) const = 0;

        template<typename T>
        bool is_a() const { return is_a(typeid(T)); }

        virtual bool equal(HolderBase const &) const = 0;

        /** return  current value */
        virtual const reference get() const = 0;

        virtual reference get() = 0;

        /** advance iterator n steps, return number of actually advanced steps  */
        virtual void advance(difference_type n = 1) = 0;

        virtual difference_type distance(HolderBase const &other) const = 0;
    };

    template<typename Tit>
    struct Holder : public HolderBase
    {
        typedef Holder<Tit> this_type;
        Tit m_iter_;
    public:

        Holder(Tit const &o_it) : m_iter_(o_it) { }

        virtual  ~Holder() final { }

        virtual std::shared_ptr<HolderBase> clone() const
        {
            return std::dynamic_pointer_cast<HolderBase>(std::make_shared<this_type>(m_iter_));
        }

        virtual bool is_a(std::type_info const &t_info) const final { return typeid(Tit) == t_info; }

        virtual bool equal(HolderBase const &other) const final
        {
            assert(other.template is_a<Tit>());
            return static_cast<Holder<Tit> const & >(other).m_iter_ == m_iter_;
        };

        virtual difference_type distance(HolderBase const &other) const final
        {
            assert(other.template is_a<Tit>());
            return std::distance(static_cast<Holder<Tit> const & >(other).m_iter_, m_iter_);
        }


        /** return  current value */
        virtual const reference get() const final { return *m_iter_; };

        virtual reference get() final { return *m_iter_; };

        /** advance iterator n steps, return number of actually advanced steps  */
        virtual void advance(difference_type n = 1) final { return std::advance(m_iter_, n); }
    };


};

class MeshEntityRange
{
    typedef MeshEntityRange this_type;

    class HolderBase;

    template<typename, bool> struct Holder;

    HAS_MEMBER_FUNCTION(range)

public :
    typedef MeshEntityIterator iterator;

    MeshEntityRange() : m_holder_(nullptr) { }

    //****************************************************************************
    // TBB Range Concept Begin
    template<typename TOther>
    MeshEntityRange(TOther const &other) :
            m_holder_(std::dynamic_pointer_cast<HolderBase>(
                    std::make_shared<Holder<TOther,
                            has_member_function_range<TOther>::value>>(other)))
    {
    }

    template<typename ...Args>
    MeshEntityRange(this_type &other, parallel::tags::split) :
            m_holder_(other.m_holder_->split())
    {
    }

    MeshEntityRange(this_type const &other) : m_holder_(
            other.m_holder_ == nullptr ? nullptr : other.m_holder_->clone()) { }

    MeshEntityRange(this_type &&other) : m_holder_(other.m_holder_) { }

    ~MeshEntityRange() { }


public:
    static const bool is_splittable_in_proportion = true;

    bool is_divisible() const { return m_holder_->is_divisible(); }

    bool empty() const { return m_holder_ == nullptr || m_holder_->empty(); }

    iterator begin() { return m_holder_->begin(); }

    iterator end() { return m_holder_->end(); }

    iterator begin() const { return m_holder_->begin(); }

    iterator end() const { return m_holder_->end(); }

    // TBB Range Concept End
    //****************************************************************************


    template<typename T, typename ...Args>
    static MeshEntityRange create(Args &&...args)
    {
        MeshEntityRange res;
        res.m_holder_ = std::dynamic_pointer_cast<HolderBase>(
                Holder<T, has_member_function_range<T>::value>::create(std::forward<Args>(args)...));
        return std::move(res);
    }

    this_type operator=(this_type const &other)
    {
        m_holder_ = other.m_holder_;
        return *this;
    }

    this_type &swap(this_type &other)
    {
        std::swap(m_holder_, other.m_holder_);
        return *this;
    }

    template<typename T> T &as()
    {
        assert(m_holder_->is_a(typeid(T)));
        return *std::dynamic_pointer_cast<Holder<T, has_member_function_range<T>::value>>(m_holder_);
    }

    template<typename T> T const &as() const
    {
        assert(m_holder_->is_a(typeid(T)));
        return *std::dynamic_pointer_cast<Holder<T, has_member_function_range<T>::value>>(m_holder_);
    }

private:

    std::shared_ptr<HolderBase> m_holder_;

    struct HolderBase
    {
        virtual std::shared_ptr<HolderBase> clone() const = 0;

        virtual std::shared_ptr<HolderBase> split() = 0;

        virtual bool is_a(std::type_info const &) const = 0;

        virtual bool is_divisible() const = 0;

        virtual bool empty() const = 0;

        virtual iterator begin() const = 0;

        virtual iterator end() const = 0;

        virtual iterator begin() = 0;

        virtual iterator end() = 0;

    };


    template<typename TOtherRange>
    struct Holder<TOtherRange, false> : public HolderBase
    {
        typedef Holder<TOtherRange, false> this_type;
        TOtherRange m_range_;
    public:

        Holder(TOtherRange const &other) : m_range_(other) { }

        template<typename ...Args>
        Holder(this_type &other, Args &&...args) : m_range_(other.m_range_, std::forward<Args>(args)...) { }

        virtual  ~Holder() { }

        template<typename ...Args>
        static std::shared_ptr<this_type> create(Args &&...args)
        {
            return std::shared_ptr<this_type>(new this_type{TOtherRange(std::forward<Args>(args)...)});
        }

        virtual std::shared_ptr<HolderBase> clone() const
        {
            return std::dynamic_pointer_cast<HolderBase>(std::make_shared<this_type>(*this));
        }

        virtual std::shared_ptr<HolderBase> split()
        {
            return std::dynamic_pointer_cast<HolderBase>(std::make_shared<this_type>(*this, parallel::tags::split()));
        };


        virtual bool is_a(std::type_info const &t_info) const final { return typeid(TOtherRange) == t_info; }

        virtual bool is_divisible() const { return m_range_.is_divisible(); }

        virtual bool empty() const { return m_range_.empty(); }

        virtual iterator begin() const { return iterator(m_range_.begin()); }

        virtual iterator end() const { return iterator(m_range_.end()); }

        virtual iterator begin() { return iterator(m_range_.begin()); }

        virtual iterator end() { return iterator(m_range_.end()); }
    };


    template<typename TContainer>
    struct Holder<TContainer, true> : public HolderBase
    {
        typedef typename TContainer::const_range_type range_type;
        typedef Holder<TContainer, true> this_type;
        std::shared_ptr<TContainer> m_container_;
        range_type m_range_;
    public:

        Holder(TContainer const &other) :
                m_container_(std::make_shared<TContainer>(other)), m_range_(m_container_->range()) { }

        template<typename ...Args>
        Holder(this_type &other, Args &&...args) :
                m_container_(other.m_container_), m_range_(other.m_range_, std::forward<Args>(args)...) { }

        virtual  ~Holder() { }

        template<typename ...Args> static std::shared_ptr<this_type>
        create(Args &&...  args)
        {
            return std::shared_ptr<this_type>(new this_type{TContainer(std::forward<Args>(args)...)});
        }

        virtual std::shared_ptr<HolderBase> clone() const
        {
            return std::dynamic_pointer_cast<HolderBase>(std::make_shared<this_type>(*this));
        }

        virtual std::shared_ptr<HolderBase> split()
        {
            return std::dynamic_pointer_cast<HolderBase>(std::make_shared<this_type>(*this, parallel::tags::split()));
        };

        virtual bool is_a(std::type_info const &t_info)
        const final { return typeid(TContainer) == t_info; }

        virtual bool is_divisible() const { return m_range_.is_divisible(); }

        virtual bool empty() const { return m_range_.empty(); }

        virtual iterator begin() const { return iterator(m_range_.begin()); }

        virtual iterator end() const { return iterator(m_range_.end()); }

        virtual iterator begin() { return iterator(m_range_.begin()); }

        virtual iterator end() { return iterator(m_range_.end()); }

    };
};

}
} //namespace simpla { namespace mesh

#endif //SIMPLA_MESH_MESHENTITY_H
