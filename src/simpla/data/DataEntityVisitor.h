//
// Created by salmon on 17-8-17.
//

#ifndef SIMPLA_DATAENTITYVISITOR_H
#define SIMPLA_DATAENTITYVISITOR_H

#include <simpla/SIMPLA_config.h>
#include <simpla/utilities/ObjectHead.h>
#include <complex>
#include <memory>

namespace simpla {
namespace data {

class DataTable;
class DataArray;
class DataBlock;

struct DataEntityVisitor {
    virtual int visit(int u) { return 0; }
    virtual int visit(Real u) { return 0; }
    virtual int visit(std::complex<Real> const& u) { return 0; }
    virtual int visit(std::string const& u) { return 0; }
    virtual int visit(int const* u, int ndims, int const* d) { return 0; }
    virtual int visit(Real const* u, int ndims, int const* d) { return 0; }
    virtual int visit(std::complex<Real> const* u, int ndims, int const* d) { return 0; }
    virtual int visit(std::string const* u, int ndims, int const* d) { return 0; }
    virtual int visit(DataTable const* u) { return 0; }
    virtual int visit(DataArray const* u) { return 0; }
    virtual int visit(DataBlock const* u) { return 0; }
};

}  // namespace data
}  // namespace simpla
#endif  // SIMPLA_DATAENTITYVISITOR_H
