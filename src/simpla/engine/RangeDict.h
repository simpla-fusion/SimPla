//
// Created by salmon on 17-7-3.
//

#ifndef SIMPLA_RANGEDICT_H
#define SIMPLA_RANGEDICT_H
namespace simpla {
namespace engine {

template <typename T>
struct RangeDict {

    template<typename ...Args>
    Range<T> & append(std::string const & g,Args &&...args)
    {

    }

};
}
}
#endif  // SIMPLA_RANGEDICT_H
