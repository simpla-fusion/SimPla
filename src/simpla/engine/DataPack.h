//
// Created by salmon on 17-7-2.
//

#ifndef SIMPLA_DATAPACK_H
#define SIMPLA_DATAPACK_H

namespace simpla {
namespace engine {
struct DataPack {
    DataPack() {}
    template <typename U>
    DataPack(U&& other) {}

    template <typename U>
    void swap(U& other) {}
    void swap(DataPack& other) {}
};
}
}

#endif  // SIMPLA_DATAPACK_H
