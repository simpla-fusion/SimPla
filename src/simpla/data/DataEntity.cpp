//
// Created by salmon on 16-6-6.
//
#include "DataEntity.h"
#include <ostream>
namespace simpla {
namespace data {
std::ostream& operator<<(std::ostream& os, DataEntity const& v) {
    v.Print(os, 0);
    return os;
}

}  // namespace data
}  // namespace simpla