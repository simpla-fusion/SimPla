/**
 * @file MeshAttribute.cpp
 * @author salmon
 * @date 2016-05-19.
 */
#include "MeshWorker.h"
#include "MeshAtlas.h"
#include "MeshAttribute.h"

#include "../data_model/DataSet.h"

namespace simpla { namespace mesh
{
data_model::DataSet MeshAttributeBase::get_dataset() const
{
    UNIMPLEMENTED;
    return data_model::DataSet();
}

void mesh::MeshAttributeBase::set_dataset(data_model::DataSet const &)
{
    UNIMPLEMENTED;
}
}}//namespace simpla{namespace mesh
