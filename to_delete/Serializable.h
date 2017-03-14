//
// Created by salmon on 16-11-2.
//

#ifndef SIMPLA_SERIALIZABLE_H
#define SIMPLA_SERIALIZABLE_H

#include <string>

namespace simpla { namespace data { class DataTable; }}

namespace simpla { namespace concept
{/**  @ingroup concept   */

/**
 * @brief a type whose instances can be converted to DataEntityTable
 * @details
 * ## Summary
 *
 * ## Requirements
 *  Class \c R implementing the concept of @ref Printable must define:
 *   Pseudo-Signature                                            | Semantics
 *	 ------------------------------------------------------------|----------
 * 	 \code   void load(data::DataEntityTable const &)   \endcode |
 * 	 \code   void save(data::DataEntityTable * )        \endcode |
 */
struct Serializable
{
    virtual void Load(data::DataTable const &) =0;

    virtual void Save(data::DataTable *) const =0;
};


}}
#endif //SIMPLA_SERIALIZABLE_H
