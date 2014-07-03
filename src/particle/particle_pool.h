/*
 * particle_pool.h
 *
 *  Created on: 2014年4月29日
 *      Author: salmon
 */

#ifndef PARTICLE_POOL_H_
#define PARTICLE_POOL_H_
#include "../utilities/log.h"
#include "../utilities/sp_type_traits.h"
#include "../utilities/container_container.h"
#include "../parallel/parallel.h"
#include "save_particle.h"

namespace simpla
{

/**
 *  @ingroup Particle
 *
 *  @brief particle container
 *
 */
template<typename TM, typename TPoint>
class ParticlePool: public ContainerContainer<typename TM::compact_index_type, TPoint>
{
	std::mutex write_lock_;

public:
	static constexpr int IForm = VERTEX;

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

private:

	bool isSorted_;

public:

	mesh_type const & mesh;
	typename mesh_type::range_type range_;
	//***************************************************************************************************
	// Constructor

	template<typename ...Others> ParticlePool(mesh_type const & pmesh, Others && ...);
	template<typename ...Others> ParticlePool(mesh_type const & pmesh, range_type const &, Others && ...);

	// Destructor
	~ParticlePool();

	void Load()
	{
	}

	template<typename TDict, typename ...Args> void Load(TDict const & dict, Args && ...others);

	std::string Save(std::string const & path) const;

	//***************************************************************************************************

	template<typename ... Args>
	auto Select(Args && ... args)
	DECL_RET_TYPE((make_mapped_range(*this, mesh.Select(range_,std::forward<Args >(args)...))))
	template<typename ... Args>
	auto Select(Args && ... args) const
	DECL_RET_TYPE((make_mapped_range(*this, mesh.Select(range_,std::forward<Args >(args)...))))

	template<typename TIterator>
	void Clear(TIterator it);

	void ClearEmpty();

	void Merge(container_type * other, container_type *dest = nullptr);

	void Add(child_container_type *src);

	template<typename TRange>
	void Remove(TRange r, child_container_type *other = nullptr);

//***************************************************************************************************

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

};

/***
 * FIXME (salmon):  We need a  thread-safe and  high performance allocator for std::map<key_type,std::list<allocator> > !!
 */
template<typename TM, typename TPoint>
template<typename ...Others>
ParticlePool<TM, TPoint>::ParticlePool(mesh_type const & pmesh, Others && ...others)
		: container_type(), mesh(pmesh), isSorted_(false), range_(pmesh.Select(IForm))
{
	Load(std::forward<Others >(others)...);
}
template<typename TM, typename TPoint>
template<typename ...Others>
ParticlePool<TM, TPoint>::ParticlePool(mesh_type const & pmesh, range_type const &range, Others && ...others)
		: container_type(), mesh(pmesh), isSorted_(false), range_(range)
{
	Load(std::forward<Others >(others)...);
}

template<typename TM, typename TPoint>
template<typename TDict, typename ...Args> void ParticlePool<TM, TPoint>::Load(TDict const & dict, Args && ...others)
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

//	ParallelDo(
//
//	[this](int t_num,int t_id)
//	{
//		container_type dest;
//		for (auto s : mesh.Select(IForm).Split(t_num,t_id))
//		{
//
// 			CHECK(mesh.Hash(s));
//			auto it = base_container_type::find(s);
//
//			if (it != base_container_type::end()) this->Sort_(&(it->second), &dest);
//		}
//		Merge(&dest);
//	}
//
//	);

	//FIXME Here should be PARALLEL (multi-thread)
	container_type dest;
	for (auto s : mesh.Select(IForm))
	{

		auto it = container_type::find(s);

		if (it != container_type::end())
			this->Sort_(&(it->second), &dest);
	}
	Merge(&dest, this);
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
template<typename TIterator>
void ParticlePool<TM, TPoint>::Clear(TIterator it)
{
	container_type::lock();
	container_type::erase(it.c_it_);
	container_type::unlock();
}

template<typename TM, typename TPoint>
void ParticlePool<TM, TPoint>::Merge(container_type * other, container_type *dest)
{
	if (dest == nullptr)
		dest = this;

	container_type::lock();
	for (auto & v : *other)
	{
		auto & c = dest->get(v.first);
		c.splice(c.begin(), v.second);
	}
	container_type::unlock();

}
template<typename TM, typename TPoint>
void ParticlePool<TM, TPoint>::Add(child_container_type* other)
{
	Sort_(other, this);
}

template<typename TM, typename TPoint>
template<typename TRange>
void ParticlePool<TM, TPoint>::Remove(TRange r, child_container_type * other)
{
	auto buffer = container_type::create_child();

	for (auto it = std::get<0>(r), ie = std::get<1>(r); it != ie; ++it)
	{
		buffer.splice(buffer.begin(), *it);
	}

	if (other != nullptr)
		other->splice(other->begin(), buffer);

}
template<typename TM, template<typename > class TModel, typename TDict, typename TPoint>
std::function<void()> CreateCommand(TModel<TM> const & model, TDict const & dict, ParticlePool<TM, TPoint> * f)
{

	if (!dict["Operation"])
	{
		PARSER_ERROR("'Operation' is not defined!");
	}

	typedef typename TM mesh_type;

	typedef typename ParticlePool<TM, TPoint>::child_container_type child_container_type;

	typedef typename ParticlePool<TM, TPoint>::particle_type particle_type;
	typedef typename ParticlePool<TM, TPoint>::particle_type particle_type;
	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef std::function<particle_type(Real, particle_type const &)> field_fun;

	auto op_ = dict.template as<field_fun>();

	auto range = f->Select(dict["Select"]);

	std::function<void()> res = [range,op_]()
	{
		for(auto & cell:range)
		{
			for(auto & p :cell)
			{
				p = op_(f->mesh.GetTime(), p);
			}

		}
	};

	return res;

}
}  // namespace simpla

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
