//
// Created by salmon on 16-12-21.
//

#ifndef SIMPLA_SHAREABLE_H
#define SIMPLA_SHAREABLE_H

#include <memory>

namespace simpla { namespace concept
{
/** @ingroup concept */

/**
 * @brief a type whose instances share ownership between multiple objects; *
 * @details
 * ## Summary
 * Requirements for a type whose instances share ownership between multiple objects;
 *
 * ## Requirements
 *  Class \c R implementing the concept of @ref Shareable must define:
 *   Pseudo-Signature                                      | Semantics
 *	 ------------------------------------------------------|----------
 * 	 \code typedef std::shared_ptr<R> RangeHolder \endcode | hold the ownership of object;
 * 	 \code private  R()                           \endcode | disable directly construct object;
 * 	 \code static RangeHolder  create()           \endcode | create an object, and return the RangeHolder
 * 	 \code RangeHolder shared_from_this()         \endcode | Returns a `RangeHolder` that shares ownership of `*this` ;
 *   \code RangeHolder shared_from_this() const   \endcode | Returns a `read-only RangeHolder` that shares `const` ownership of `*this` ;
 *
 */
struct Shareable
{
    typedef std::shared_ptr<Shareable> Holder;

    virtual Holder create()=0;

    virtual Holder shared_from_this()=0;

    virtual Holder shared_from_this() const = 0;
};

}}//namespace simpla{namespace concept{
#endif //SIMPLA_SHAREABLE_H
