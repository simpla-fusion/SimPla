/**
 * @file iterator.h
 *
 * @date 2014-7-4
 * @author salmon
 */

#ifndef SP_ITERATOR_H_
#define SP_ITERATOR_H_
#include "../type_traits.h"
#include <iterator>
namespace simpla
{

/**
 * @ingroup gtl
 * @addtogroup  iterator
 * @{
 *
 */

template<typename TIterator, typename TPred = std::nullptr_t,
		typename TPolicy = std::nullptr_t, bool IsReferenceStorage = false> class Iterator;

template<typename Policy, typename TIterator, typename TPred>
auto make_iterator(TIterator const &k_ib, TIterator const & k_ie, TPred && pred)
DECL_RET_TYPE ((Iterator<
				typename std::remove_reference<TIterator>::type,
				typename std::remove_reference<TPred>::type,
				Policy,
				!std::is_lvalue_reference<TPred>::value
				>(k_ib, k_ie, std::forward<TPred>(pred))))

template<typename Policy, typename TRange, typename ...Others>
auto make_range(TRange const& r,
		Others &&... others)
				DECL_RET_TYPE ((std::make_pair(make_iterator<Policy >( begin(r), end(r), std::forward<Others>(others)...),
										make_iterator<Policy >( end(r),end(r), std::forward<Others>(others)...))))
} // namespace simpla

namespace std
{

template<typename TI, typename TPred, typename Policy>
struct iterator_traits<simpla::Iterator<TI, TPred, Policy>>
{
	typedef simpla::Iterator<TI, TPred> iterator;
	typedef typename iterator::iterator_category iterator_category;
	typedef typename iterator::value_type value_type;
	typedef typename iterator::difference_type difference_type;
	typedef typename iterator::pointer pointer;
	typedef typename iterator::reference reference;

};
/** @}*/
}  // namespace std

#endif /* SP_ITERATOR_H_ */
