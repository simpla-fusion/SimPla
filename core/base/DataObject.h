/**
 * @file data_object.h
 * @author salmon
 * @date 2015-12-16.
 */

#ifndef SIMPLA_DATA_OBJECT_H
#define SIMPLA_DATA_OBJECT_H


#include "../data_model/DataSet.h"
#include "../gtl/Properties.h"
#include "Object.h"

namespace simpla { namespace base
{

class DataObject : public Object
{
public:

    SP_OBJECT_HEAD(DataObject, Object);


    virtual Properties &properties() = 0;

    virtual Properties const &properties() const = 0;

    virtual data_model::DataSet data_set() const = 0;

    virtual std::ostream &print(std::ostream &os, int indent = 0) const;


};


}}//namespace simpla { namespace base

#endif //SIMPLA_DATA_OBJECT_H
