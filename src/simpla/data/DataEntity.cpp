//
// Created by salmon on 16-6-6.
//
#include "DataEntity.h"
#include <ostream>
namespace simpla {
namespace data {
std::ostream& operator<<(std::ostream& os, DataEntity const&) { return os; }

}  // namespace data
}  // namespace simpla