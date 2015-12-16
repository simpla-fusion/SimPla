/**
 * @file attribute.h
 * @author salmon
 * @date 2015-12-16.
 */

#ifndef SIMPLA_ATTRIBUTE_H
#define SIMPLA_ATTRIBUTE_H

#include "dataset.h"

namespace simpla
{
class AttributeBase
{
public:

    virtual DataSet dataset() const = 0;

    virtual int center_type() const = 0;

    virtual int rank() const = 0;

    virtual int extent(int i) const = 0;

};
}//namespace simpla { namespace data_model

#endif //SIMPLA_ATTRIBUTE_H
