/*
 * particle_pool.h
 *
 *  Created on: 2014-4-29
 *      Author: salmon
 */

#ifndef PARTICLE_POOL_H_
#define PARTICLE_POOL_H_
#include "../utilities/log.h"
#include "../utilities/sp_type_traits.h"
#include "../utilities/container_container.h"
#include "../utilities/sp_iterator_mapped.h"
#include "../parallel/parallel.h"
#include "save_particle.h"

namespace simpla
{

/**
 *  \ingroup Particle
 *
 *  \brief particle container
 *
 */
template<typename TM, typename TPoint>
class ParticlePool: public ContainerContainer<typename TM::compact_index_type, TPoint>
{
	std::mutex write_lock_;

public:
	static constexpr  unsigned int  IForm = VERTEX;

	typedef TM mesh_type;

	typedef TPoint particle_type;

	typedef typename TM::compact_index_type key_type;

	typedef ParticlePool<mesh_type, particle_type> this_type;

	typedef ContainerContainer<key_type, particle_type> container_type;

	typedef typename container_type::value_type child_container_type;

	typedef typename mesh_type::iterator mesh_iterator;

	typedef typename mesh_type::scalar_type scalar_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef typename mesh_type::range_type range_type;

	typedef typename particle_type::compact_type compact_particle_type;

	typedef std::function<compact_particle_type(compact_particle_type)> pointwise_fun_type;

private:

	bool isSorted_;

public:

	mesh_type const & mesh;

	// Constructor
	ParticlePool(mesh_type const & pmesh);

	// Destructor
	~ParticlePool();

	void Load(std::string const & path)
	{
		//@todo load data from file
	}

	std::string Save(std::string const & path) const;

	void ClearEmpty();

	void Add(container_type * other);

	void Add(child_container_type *src);

	template<typename TRange>
	void Remove(TRange const & range, child_container_type *other = nullptr);

	template<typename TRange, typename TFun>
	void Remove(TRange const & range, TFun const & fun, child_container_type * other = nullptr);

	template<typename TRange, typename TFun>
	void Modify(TRange const & range, TFun const & fun);

	void Sort();

	bool is_sorted() const
	{
		return isSorted_;
	}

	void EnableSort()
	{
		isSorted_ = false;
	}

private:
	/**
	 *  resort particles in cell 's', and move out boundary particles to 'dest' container
	 * @param
	 */
	template<typename TSrc, typename TDest> void Sort_(TSrc *, TDest *dest);

}
;

/***
 * @todo (salmon):  We need a  thread-safe and  high performance allocator for std::map<key_type,std::list<allocator> > !!
 */
template<typename TM, typename TPoint>
ParticlePool<TM, TPoint>::ParticlePool(mesh_type const & pmesh)
		: container_type(), mesh(pmesh), isSorted_(false)
{
}

template<typename TM, typename TPoint>
ParticlePool<TM, TPoint>::~ParticlePool()
{
}

template<typename TM, typename TPoint>
std::string ParticlePool<TM, TPoint>::Save(std::string const & name) const
{
	return simpla::Save(name, *this);
}

template<typename TM, typename TPoint>
template<typename TSrc, typename TDest>
void ParticlePool<TM, TPoint>::Sort_(TSrc * p_src, TDest *p_dest_contianer)
{

	auto pt = p_src->begin();

	auto shift = mesh.GetShift(IForm);

	while (pt != p_src->end())
	{
		auto p = pt;
		++pt;

		auto id = mesh.CoordinatesGlobalToLocal((p->x), shift);
		p->x = mesh.CoordinatesLocalToGlobal(std::get<0>(id), std::get<1>(id));
		auto & dest = p_dest_contianer->get(std::get<0>(id));
		dest.splice(dest.begin(), *p_src, p);

	}

}

template<typename TM, typename TPoint>
void ParticlePool<TM, TPoint>::Sort()
{

	if (is_sorted())
		return;

	//@bug Here should be PARALLEL (multi-threads)
	container_type dest;
	for (auto s : mesh.Select(IForm))
	{

		auto it = container_type::find(s);

		if (it != container_type::end())
			this->Sort_(&(it->second), &dest);
	}
	Add(&dest);
	isSorted_ = true;

}

template<typename TM, typename TPoint>
void ParticlePool<TM, TPoint>::ClearEmpty()
{
	container_type::lock();
	auto it = container_type::begin(), ie = container_type::end();

	while (it != ie)
	{
		auto t = it;
		++it;
		if (t->second.empty())
		{
			container_type::erase(t);
		}
	}
	container_type::unlock();
}

template<typename TM, typename TPoint>
void ParticlePool<TM, TPoint>::Add(container_type * other)
{

	container_type::lock();
	for (auto & v : *other)
	{
		auto & c = this->get(v.first);
		c.splice(c.begin(), v.second);
	}
	container_type::unlock();
	Sort();

}
template<typename TM, typename TPoint>
void ParticlePool<TM, TPoint>::Add(child_container_type* other)
{
	Sort_(other, this);
}

template<typename TM, typename TPoint>
template<typename TRange>
void ParticlePool<TM, TPoint>::Remove(TRange const & r, child_container_type * other)
{
	child_container_type buffer = container_type::create_child();

	for (auto s : r)
	{
		auto cell_it = container_type::find(s);

		if (cell_it == container_type::end())
			continue;

		buffer.splice(buffer.begin(), cell_it->second);

		container_type::erase(cell_it);
	}

	if (other != nullptr)
		other->splice(other->begin(), buffer);

}
template<typename TM, typename TPoint>
template<typename TRange, typename TFun>
void ParticlePool<TM, TPoint>::Remove(TRange const & range, TFun const & obj, child_container_type * other)
{
	std::function<bool(compact_particle_type)> fun = TypeCast<std::function<bool(compact_particle_type)>>(obj);

	auto buffer = container_type::create_child();

	for (auto s : range)
	{
		auto cell_it = container_type::find(s);

		if (cell_it == container_type::end())
			continue;

		auto it = cell_it->second.begin();
		auto ie = cell_it->second.end();

		do
		{
			auto it_p = it;
			++it;

			if (fun(particle_type::Compact(*it_p)))
			{
				buffer.splice(buffer.begin(), cell_it->second, it_p);
			}

		} while (it != ie);

	}
	if (other != nullptr)
		other->splice(other->begin(), buffer);

}

template<typename TM, typename TPoint> template<typename TRange, typename TFun>
void ParticlePool<TM, TPoint>::Modify(TRange const & range, TFun const & obj)
{
	pointwise_fun_type fun = TypeCast<pointwise_fun_type>(obj);

	size_t count = 0;
	for (auto s : range)
	{
		auto it = container_type::find(s);
		if (it != container_type::end())
		{
			for (auto & p : it->second)
			{
				p = particle_type::Decompact(fun(particle_type::Compact(p)));
			}
			++count;
		}
	}
	if (count > 0)
		isSorted_ = false;

}

}
// namespace simpla

//private:
//
//	template<bool IsConst>
//	struct cell_iterator_
//	{
//
//		typedef cell_iterator_<IsConst> this_type;
//
//		/// One of the @link iterator_tags tag types@endlink.
//		typedef std::forward_iterator_tag iterator_category;
//
//		/// The type "pointed to" by the iterator.
//		typedef typename std::conditional<IsConst, const cell_type, cell_type>::type value_type;
//
//		/// This type represents a pointer-to-value_type.
//		typedef value_type * pointer;
//
//		/// This type represents a reference-to-value_type.
//		typedef value_type & reference;
//
//		typedef typename std::conditional<IsConst, container_type const &, container_type&>::type container_reference;
//
//		typedef typename std::conditional<IsConst, typename container_type::const_iterator,
//				typename container_type::iterator>::type cell_iterator;
//
//		container_reference data_;
//		key_type m_it_, m_ie_;
//		cell_iterator c_it_;
//
//		cell_iterator_(container_reference data, mesh_iterator m_ib, mesh_iterator m_ie) :
//				data_(data), m_it_(m_ib), m_ie_(m_ie), c_it_(base_container_type::find(m_ib))
//		{
//			UpldateCellIteraor_();
//		}
//
//		~cell_iterator_()
//		{
//		}
//
//		reference operator*()
//		{
//			return c_it_->second;
//		}
//		const reference operator*() const
//		{
//			return c_it_->second;
//		}
//		pointer operator ->()
//		{
//			return &(c_it_->second);
//		}
//		const pointer operator ->() const
//		{
//			return &(c_it_->second);
//		}
//		bool operator==(this_type const & rhs) const
//		{
//			return m_it_ == rhs.m_it_;
//		}
//
//		bool operator!=(this_type const & rhs) const
//		{
//			return !(this->operator==(rhs));
//		}
//
//		this_type & operator ++()
//		{
//			++m_it_;
//			UpldateCellIteraor_();
//			return *this;
//		}
//		this_type operator ++(int)
//		{
//			this_type res(*this);
//			++res;
//			return std::move(res);
//		}
//	private:
//		void UpldateCellIteraor_()
//		{
//			c_it_ = base_container_type::find(m_it_);
//			while (c_it_ == base_container_type::end() && m_it_ != m_ie_)
//			{
//				++m_it_;
//				c_it_ = base_container_type::find(m_it_);
//			}
//		}
//	};
//
//	template<bool IsConst>
//	struct cell_range_: public mesh_type::range
//	{
//
//		typedef cell_iterator_<IsConst> iterator;
//		typedef cell_range_<IsConst> this_type;
//		typedef typename mesh_type::range mesh_range;
//		typedef typename std::conditional<IsConst, const container_type &, container_type&>::type container_reference;
//
//		container_reference data_;
//
//		cell_range_(container_reference data, mesh_range const & m_range) :
//				mesh_range(m_range), data_(data)
//		{
//		}
//
//		template<typename ...Args>
//		cell_range_(container_reference data,Args && ...args) :
//				mesh_range(std::forward<Args >(args)...), data_(data)
//		{
//		}
//
//		~cell_range_()
//		{
//		}
//
//		iterator begin() const
//		{
//			return iterator(data_, mesh_range::begin(), mesh_range::end());
//		}
//
//		iterator end() const
//		{
//
//			return iterator(data_, mesh_range::end(), mesh_range::end());
//		}
//
//		iterator rbegin() const
//		{
//			return iterator(data_, mesh_range::begin(), mesh_range::end());
//		}
//
//		iterator rend() const
//		{
//			return iterator(data_, mesh_range::rend(), mesh_range::end());
//		}
//
//		template<typename ...Args>
//		this_type Split(Args const & ... args) const
//		{
//			return this_type(data_, mesh_range::Split(std::forward<Args >(args)...));
//		}
//
//		template<typename ...Args>
//		this_type SubRange(Args const & ... args) const
//		{
//			return this_type(data_, mesh_range::SubRange(std::forward<Args >(args)...));
//		}
//		size_t size() const
//		{
//			size_t count = 0;
//			for (auto it = begin(), ie = end(); it != ie; ++it)
//			{
//				++count;
//			}
//			return count;
//		}
//	};
//	template<typename TV>
//	struct iterator_
//	{
//
//		typedef iterator_<TV> this_type;
//
///// One of the @link iterator_tags tag types@endlink.
//		typedef std::bidirectional_iterator_tag iterator_category;
//
///// The type "pointed to" by the iterator.
//		typedef particle_type value_type;
//
///// This type represents a pointer-to-value_type.
//		typedef value_type* pointer;
//
///// This type represents a reference-to-value_type.
//		typedef value_type& reference;
//
//		typedef cell_iterator_<value_type> cell_iterator;
//
//		typedef typename std::conditional<std::is_const<TV>::value, typename cell_iterator::value_type::const_iterator,
//		        typename cell_iterator::value_type::iterator>::type element_iterator;
//
//		cell_iterator c_it_;
//		element_iterator e_it_;
//
//		template<typename ...Args>
//		iterator_(Args const & ...args)
//				: c_it_(std::forward<Args >(args)...), e_it_(c_it_->begin())
//		{
//
//		}
//		iterator_(cell_iterator c_it, element_iterator e_it)
//				: c_it_(c_it), e_it_(e_it)
//		{
//
//		}
//		~iterator_()
//		{
//		}
//
//		reference operator*() const
//		{
//			return *e_it_;
//		}
//
//		pointer operator ->() const
//		{
//			return &(*e_it_);
//		}
//		bool operator==(this_type const & rhs) const
//		{
//			return c_it_ == rhs.c_it_ && (c_it_.isNull() || e_it_ == rhs.e_it_);
//		}
//
//		bool operator!=(this_type const & rhs) const
//		{
//			return !(this->operator==(rhs));
//		}
//
//		this_type & operator ++()
//		{
//			if (!c_it_.isNull())
//			{
//
//				if (e_it_ != c_it_->end())
//				{
//					++e_it_;
//
//				}
//				else
//				{
//					++c_it_;
//
//					if (!c_it_.isNull())
//						e_it_ = c_it_->begin();
//				}
//
//			}
//
//			return *this;
//		}
//		this_type operator ++(int)
//		{
//			this_type res(*this);
//			++res;
//			return std::move(res);
//		}
//		this_type & operator --()
//		{
//			if (!c_it_.isNull())
//			{
//
//				if (e_it_ != c_it_->rend())
//				{
//					--e_it_;
//
//				}
//				else
//				{
//					--c_it_;
//
//					if (!c_it_.isNull())
//						e_it_ = c_it_.rbegin();
//				}
//
//			}
//			return *this;
//		}
//		this_type operator --(int)
//		{
//			this_type res(*this);
//			--res;
//			return std::move(res);
//		}
//	};
//	template<typename TV>
//	struct range_: public cell_range_<TV>
//	{
//
//		typedef iterator_<TV> iterator;
//		typedef range_<TV> this_type;
//		typedef cell_range_<TV> cell_range_type;
//		typedef typename std::conditional<std::is_const<TV>::value, const container_type &, container_type&>::type container_reference;
//
//		range_(cell_range_type range)
//				: cell_range_type(range)
//		{
//		}
//		template<typename ...Args>
//		range_(container_reference d,Args && ... args)
//				: cell_range_type(d, std::forward<Args >(args)...)
//		{
//		}
//
//		~range_()
//		{
//		}
//
//		iterator begin() const
//		{
//			auto cit = cell_range_type::begin();
//			return iterator(cit, cit->begin());
//		}
//
//		iterator end() const
//		{
//			auto cit = cell_range_type::rbegin();
//			return iterator(cit, cit->end());
//		}
//
//		iterator rbegin() const
//		{
//			auto cit = cell_range_type::rbegin();
//			return iterator(cit, cit->rbegin());
//		}
//
//		iterator rend() const
//		{
//			auto cit = cell_range_type::begin();
//			return iterator(cit, cit->rend());
//		}
//
//		size_t size() const
//		{
//			size_t count = 0;
//			for (auto it = cell_range_type::begin(), ie = cell_range_type::end(); it != ie; ++it)
//			{
//				count += it->size();
//			}
//			return count;
//		}
//		template<typename ...Args>
//		this_type Split(Args const & ... args) const
//		{
//			return this_type(cell_range_type::Split(std::forward<Args >(args)...));
//		}
//	};
//	typedef iterator_<particle_type> iterator;
//	typedef iterator_<const particle_type> const_iterator;
//
//	typedef cell_iterator_<false> cell_iterator;
//	typedef cell_iterator_<true> const_cell_iterator;

//	typedef range_<particle_type> range;
//	typedef range_<const particle_type> const_range;

//	iterator begin()
//	{
//		return iterator(base_container_type::begin());
//	}
//
//	iterator end()
//	{
//		return iterator(base_container_type::rbegin(), base_container_type::rbegin()->end());
//	}
//
//	iterator rbegin()
//	{
//		return iterator(base_container_type::rbegin(), base_container_type::rbeing()->rbegin());
//	}
//
//	iterator rend()
//	{
//		return iterator(base_container_type::begin(), base_container_type::begin()->rend());
//	}
//
//	typename container_type::iterator cell_begin()
//	{
//		return (base_container_type::begin());
//	}
//
//	typename container_type::iterator cell_end()
//	{
//		return (base_container_type::end());
//	}
//	typename container_type::iterator cell_rbegin()
//	{
//		return (base_container_type::rbegin());
//	}
//
//	typename container_type::iterator cell_rend()
//	{
//		return (base_container_type::rend());
//	}
//	typename container_type::const_iterator cell_begin() const
//	{
//		return (base_container_type::cbegin());
//	}
//
//	typename container_type::const_iterator cell_end() const
//	{
//		return (base_container_type::cend());
//	}
//
//	typename container_type::const_iterator cell_rbegin() const
//	{
//		return (base_container_type::crbegin());
//	}
//
//	typename container_type::const_iterator cell_rend() const
//	{
//		return (base_container_type::crend());
//	}
#endif /* PARTICLE_POOL_H_ */
