/*
 * @file enable_create_from_this.h
 *
 *  Created on: 2015年1月15日
 *      Author: salmon
 */

#ifndef CORE_GTL_ENABLE_CREATE_FROM_THIS_H_
#define CORE_GTL_ENABLE_CREATE_FROM_THIS_H_
#include <memory>

namespace simpla
{

//#ifdef USE_TBB
//#include <tbb/tbb.h>
//typedef tbb::split op_split;
//#else
class op_split
{
};
//#endif

class op_clone
{
};

class op_select
{
};
class op_merge
{
};
/**
 *  @ingroup gtl
 *  @[
 *  @addtogroup concept Concept
 * @}
 */
/**
 *  @ingroup   concept
 *  @addtogroup shareable Shareable
 *  @{
 * ## Summary
 * Requirements for a type whose instances share ownership between multiple objects;
 *
 * ## Requirements
 * For @ref shareable Object `X`
 *
 *   Pseudo-Signature                    | Semantics
 *	 ------------------------------------|----------
 * 	 `typedef std::shared_ptr<X> holder` | hold the ownership of object;
 * 	 `private  X()`                      | disable directly construct object;
 * 	 `static holder  create()`           | create an object, and return the holder
 * 	 `holder shared_from_this()`         | Returns a `holder` that shares ownership of `*this` ;
 *   `holder shared_from_this() const`   | Returns a `read-only holder` that shares `const` ownership of `*this` ;
 *
 *  @}
 */
/**
 *  @ingroup   concept
 *  @addtogroup splittable Splitable
 *  @{
 * ## Summary
 *  Requirements for a type whose instances can be split into two pieces.
 *
 *  @ref tbb::splittable
 *
 *  @ref https://software.intel.com/zh-cn/node/506141
 *
 * ## Requirements
 *
 *	Pseudo-Signature   |Semantics
 * ------------------- |----------
 * `X::X(X& x, split)` | Split x into x and newly constructed object.
 *
 * ## Description
 * > _from TBB_
 * >
 * >   A type is splittable if it has a splitting constructor that allows
 * >  an instance to be split into two pieces. The splitting constructor
 * >  takes as arguments a reference to the original object, and a dummy
 * >   argument of type split, which is defined by the library. The dummy
 * >   argument distinguishes the splitting constructor from a copy constructor.
 * >   After the constructor runs, x and the newly constructed object should
 * >   represent the two pieces of the original x. The library uses splitting
 * >    constructors in two contexts:
 * >    - Partitioning a range into two subranges that can be processed concurrently.
 * >    - Forking a body (function object) into two bodies that can run concurrently.
 *
 * - Split  @ref Container  into two part, that can be accessed concurrently
 * - if X::left() and X::right() exists, construct proportion with the
 * ratio specified by left() and right(). (alter from tbb::proportional_split)
 *
 * ## Description
 *
 *
 *@}
 */
/**
 *  @ingroup gpl
 *
 *  @brief manage ownership of object that 'generate' from other object.
 *
 * ## Summary
 *   ` enable_create_from_this` allows an object `t` that is currently
 *   managed by a `std::shared_ptr` named `pt` to safely generate additional
 *    `std::shared_ptr` instances `pt1, pt2, ...` that generator from `pt` and
 *    all share ownership of `t` with `pt`.
 *
 * ## Requirement
 *   - require `Object::Object(Object & ,...)` exists
 *
 * ## Usage
 *
 * @code{
 *
 *     struct X:public Holder<X>
 *     {
 *       X(args...);
 *       ....
 *     };
 *
 *     auto x= X::create(args...)
 *     auto y1= x->create_from_this();
 *     auto y2= y1->create_from_this(split());
 *     auto y3= t2->create_from_this(split());
 *
 *     assert(y2->root_holder()==  y3->root_holder());
 *
 *
 * @endcode}
 *
 */
template<typename TObject>
struct enable_create_from_this: public std::enable_shared_from_this<TObject>
{

	typedef TObject object_type;
	typedef enable_create_from_this<object_type> this_type;

	typedef std::shared_ptr<object_type> holder;
	typedef std::shared_ptr<const object_type> const_holder;

	holder root_;

	enable_create_from_this() :
			root_(nullptr)
	{
	}
	enable_create_from_this(this_type & other) :
			root_(other.root_)
	{
	}
	virtual ~enable_create_from_this()
	{

	}
	virtual object_type & self()=0;
	virtual object_type const & self() const =0;

	/**
	 *  alias of `std::make_shared<object_type>`
	 * @param args
	 * @return
	 */
	template<typename ...Args>
	static holder create(Args && ...args)
	{
		return std::make_shared<object_type>(std::forward<Args>(args)...);
	}

	static holder create()
	{
		return holder(new object_type());
	}

	template<typename TOther, typename ...Args>
	std::shared_ptr<TOther> create_from_this(Args && ...args)
	{
		static_assert( std::is_base_of<object_type,TOther>::value,
				"this is base of TOther");

		auto res = std::make_shared<object_type>(self(),
				std::forward<Args>(args)...);
		res->root_ = root_holder();
		return std::move(res);
	}

	template<typename TOther, typename ...Args>
	std::shared_ptr<TOther> create_from_this(Args && ...args) const
	{
		static_assert( std::is_base_of<object_type,TOther>::value,
				"this is base of TOther");

		auto res = std::make_shared<object_type>(self(),
				std::forward<Args>(args)...);
		res->root_ = root_holder();
		return std::move(res);
	}

	using std::enable_shared_from_this<object_type>::shared_from_this;

	holder split_from_this()
	{
		return std::move(create_from_this(self(), op_split()));
	}

	template<typename ...Args>
	holder split_from_this(Args && ...args)
	{
		return std::move(
				create_from_this<object_type>(self(), op_split(),
						std::forward<Args>(args)...));
	}

//	template<typename ...Args>
//	holder select_from_this(Args && ...args)
//	{
//		return std::move(
//				create_from_this<object_type>(self(), op_select(),
//						std::forward<Args>(args)...));
//	}
//
//	template<typename ...Args>
//	holder select_from_this(Args && ...args) const
//	{
//		return std::move(
//				create_from_this(self(), op_select(),
//						std::forward<Args>(args)...));
//	}
//
//	holder merge_with_this(object_type && ...args) const
//	{
//		return std::move(
//				create_from_this(self(), op_merge(),
//						std::forward<object_type>(args)...));
//	}
//	holder merge_with_this(object_type && ...args)
//	{
//		return std::move(
//				create_from_this(self(), op_merge(),
//						std::forward<object_type>(args)...));
//	}
	holder root_holder()
	{
		return (root_ == nullptr) ? shared_from_this() : root_;
	}

	holder root_holder() const
	{
		return (root_ == nullptr) ? shared_from_this() : root_;
	}

	object_type & root()
	{
		return (root_ == nullptr) ? self() : *root_;
	}

	object_type const & root() const
	{
		return (root_ == nullptr) ? self() : *root_;
	}

	bool is_root() const
	{
		return root_ == nullptr;
	}
};

}  // namespace simpla

#endif /* CORE_GTL_ENABLE_CREATE_FROM_THIS_H_ */
