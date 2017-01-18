//
// Created by salmon on 16-9-27.
//

#ifndef SIMPLA_REMOTEDOMAIN_H
#define SIMPLA_REMOTEDOMAIN_H

namespace simpla
{
namespace io { struct IOStream; }
namespace parallel { struct DistributedObject; }

namespace mesh
{
class Mesh;

class DataBlock;
} //namespace get_mesh
} //namespace simpla

namespace simpla { namespace simulation
{

class VirtualDomain : public ProblemDomain
{

};
}}//namespace simpla { namespace simulation

#endif //SIMPLA_REMOTEDOMAIN_H
