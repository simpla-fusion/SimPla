/**
 * @file block_connection.h
 * Created by salmon on 7/8/15.
 */
#ifndef SIMPLA_BLOCK_CONNECTION_H
#define SIMPLA_BLOCK_CONNECTION_H

#include <type_traits>
#include "../mesh_layout.h"

namespace simpla {


namespace tags { template<int LEVEL> struct multi_block; }

template<typename CS, int LEVEL>
struct MeshConnection<Mesh < CS, tags::multi_block<LEVEL>>, Mesh <CS, tags::multi_block<LEVEL>> > {

MeshConnection()
{ }

virtual ~MeshConnection()
{ }

};
}
#endif //SIMPLA_BLOCK_CONNECTION_H
