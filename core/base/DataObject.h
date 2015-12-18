/**
 * @file DataObject.h
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

class DataObject : public SpObject
{
public:

//    virtual std::shared_ptr<DataObject> clone() const { return std::make_shared<DataObject>(); }

    SP_OBJECT_HEAD(DataObject, SpObject);

    virtual data_model::DataSet data_set() = 0;

    virtual data_model::DataSet data_set() const = 0;

    inline Properties &properties(std::string const &key)
    {
        this->touch();
        return m_properties_[key];
    }

    inline Properties const &properties(std::string const &key) const { return m_properties_.at(key); }

    inline Properties const &properties() const { return m_properties_; }

private:
    Properties m_properties_;


};

#define SP_OBJECT_PROPERTIES(_TYPE_, _NAME_)                                  \
void _NAME_(_TYPE_ const & v){ this->properties(__STRING(_NAME_)) = v;}         \
_TYPE_ _NAME_()const{return properties(__STRING(_NAME_)).as<_TYPE_>();}                             \



}}//namespace simpla { namespace base

#endif //SIMPLA_DATA_OBJECT_H
