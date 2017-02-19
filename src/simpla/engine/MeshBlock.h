//
// Created by salmon on 17-2-13.
//

#ifndef SIMPLA_MESHBLOCK_H
#define SIMPLA_MESHBLOCK_H

#include <simpla/SIMPLA_config.h>
#include <simpla/toolbox/sp_def.h>
namespace simpla {
namespace engine {
class MeshBlock {
   public:
    template <typename... Args>
    MeshBlock(Args &&... args) {}
    ~MeshBlock() {}
    size_tuple dimensions() const { return size_tuple{1, 1, 1}; }
    void id(id_type) {}
    id_type id() const { return NULL_ID; }
    template <typename U>
    size_type hash(U const &) const {
        return 0;
    }
    point_type dx() const { return point_type{}; }
    template <typename... Args>
    point_type point(Args &&... args) const {
        return point_type{};
    }
};
}
}
#endif  // SIMPLA_MESHBLOCK_H
