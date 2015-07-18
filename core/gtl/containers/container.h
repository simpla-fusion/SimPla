/**
 * @file container.h
 *
 *  Created on: 2014-12-30
 *      Author: salmon
 */

#ifndef CORE_CONTAINERS_CONTAINER_H_
#define CORE_CONTAINERS_CONTAINER_H_

/**
 *  @ingroup gtl
 *  @addtogroup container Container
 *  @{
 * ## Summary
 * - @ref container is @ref splittable
 * - @ref container is @ref shareable
 *
 * ## Requirements

 *   For @ref splittable @ref container `X`
 *
 *    Pseudo-Signature                    | Semantics
 *	 ------------------------------------|----------
 * 	 `holder split_from_this(...)`       | Split *this into two pieces, that use *this as _root_. One replaces *this and  another one is returned.
 * 	 `const_holder root()`                     | Return 'root container' that reference to whole data. If `*this` is never split , return `*this`
 *
 * ## Description
 *
 *  @}
 */

#endif /* CORE_CONTAINERS_CONTAINER_H_ */
