/**
 * @file data_object.h
 * @author salmon
 * @date 2015-12-16.
 */

#ifndef SIMPLA_DATA_OBJECT_H
#define SIMPLA_DATA_OBJECT_H


#include "../data_model/DataSet.h"
#include "../toolbox/Properties.h"
#include "Object.h"

namespace simpla { namespace base
{

class DataObject
{
public:


    virtual Properties &properties() = 0;

    virtual Properties const &properties() const = 0;

    virtual data_model::DataSet data_set() const = 0;

    virtual std::ostream &print(std::ostream &os, int indent = 1) const;

};


}}//namespace simpla { namespace base

#endif //SIMPLA_DATA_OBJECT_H
