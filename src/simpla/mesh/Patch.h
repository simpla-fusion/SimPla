//
// Created by salmon on 16-11-2.
//

#ifndef SIMPLA_PATCH_H
#define SIMPLA_PATCH_H

#include <simpla/SIMPLA_config.h>
#include <simpla/data/Serializable.h>

namespace simpla { namespace mesh
{

struct PatchBase : public data::Serializable
{

};

template<typename V, typename M, size_type IFORM>
struct Patch : public PatchBase
{

};
}}
#endif //SIMPLA_PATCH_H
