//
// Created by salmon on 16-11-17.
//

#ifndef SIMPLA_CONFIGURABLE_H
#define SIMPLA_CONFIGURABLE_H

#include <simpla/data/DataEntityTable.h>

namespace simpla { namespace concept
{
/**  @ingroup concept   */


/**
 * @brief a type whose instances has member DataEntityTable db; *
 * @details
 * ## Summary
 * Requirements for a type whose instances share ownership between multiple objects;
 *
 * ## Requirements
 *  Class \c R implementing the concept of @ref Configurable must define:
 *   Pseudo-Signature                                      | Semantics
 *	 ------------------------------------------------------|----------
 * 	 \code   R()                                  \endcode | constructor;
 * 	 \code  virtual ~R()                          \endcode | Destructor
 * 	 \code data::DataEntityTable db               \endcode |
 *   \code std::string name() const               \endcode | if key-value 'name' return it else return empty string
 *
 */

struct Configurable
{
    data::DataEntityTable db;

    Configurable() {}

    virtual ~Configurable() {}

    std::string name() const { return db.get_value("name", std::string("")); }

};


}} // namespace  simpla::concept


#endif //SIMPLA_CONFIGURABLE_H
