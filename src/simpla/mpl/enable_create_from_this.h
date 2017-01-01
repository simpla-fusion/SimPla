/**
 * @file enable_create_from_this.h
 *
 *  Created on: 2015-1-15
 *      Author: salmon
 */

#ifndef CORE_toolbox_ENABLE_CREATE_FROM_THIS_H_
#define CORE_toolbox_ENABLE_CREATE_FROM_THIS_H_
#include <memory>

namespace simpla
{



/**
 *  @ingroup gpl
 *
 *  @brief manage ownership of object that 'generate' from other object.
 *
 * ## Summary
 *   ` enable_create_from_this` allows an object `t` that is currently
 *   managed by a `std::shared_ptr` named `pt` to safely generate additional
 *    `std::shared_ptr` instances `pt1, pt2, ...` that generate from `pt` and
 *    all share ownership of `t` with `pt`.
 *
 * ## Requirement
 *   - require `GeoObject::GeoObject(GeoObject & ,...)` exists
 *
 * ## Usage
 *
 * @code{
 *
 *     struct X:public RangeHolder<X>
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

	enable_create_from_this()
			: root_(nullptr)
	{
	}
	enable_create_from_this(this_type & other)
			: root_(other.root_)
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
				"this is PlaceHolder of TOther");

		auto res = std::make_shared<object_type>(self(),
				std::forward<Args>(args)...);
		res->root_ = root_holder();
		return std::move(res);
	}

	template<typename TOther, typename ...Args>
	std::shared_ptr<TOther> create_from_this(Args && ...args) const
	{
		static_assert( std::is_base_of<object_type,TOther>::value,
				"this is PlaceHolder of TOther");

		auto res = std::make_shared<object_type>(self(),
				std::forward<Args>(args)...);
		res->root_ = root_holder();
		return std::move(res);
	}

	using std::enable_shared_from_this<object_type>::shared_from_this;

//	template<typename ...Args>
//	Holder select_from_this(Args && ...args)
//	{
//		return std::move(
//				create_from_this<object_type>(self(), op_select(),
//						std::forward<Args>(args)...));
//	}
//
//	template<typename ...Args>
//	Holder select_from_this(Args && ...args) const
//	{
//		return std::move(
//				create_from_this(self(), op_select(),
//						std::forward<Args>(args)...));
//	}
//
//	Holder merge_with_this(object_type && ...args) const
//	{
//		return std::move(
//				create_from_this(self(), op_merge(),
//						std::forward<object_type>(args)...));
//	}
//	Holder merge_with_this(object_type && ...args)
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

#endif /* CORE_toolbox_ENABLE_CREATE_FROM_THIS_H_ */
