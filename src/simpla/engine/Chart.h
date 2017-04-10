//
// Created by salmon on 17-2-17.
//

#ifndef SIMPLA_CHART_H
#define SIMPLA_CHART_H

#include <simpla/concept/Configurable.h>
#include <simpla/geometry/GeoObject.h>
#include <memory>
#include "simpla/SIMPLA_config.h"
#include "simpla/data/DataTable.h"
#include "simpla/toolbox/sp_def.h"

namespace simpla {
namespace engine {
class Mesh;
class MeshBlock;
class Chart : public concept::Configurable {
   public:
    Chart();
    virtual ~Chart();

    virtual Mesh *CreateView(std::shared_ptr<MeshBlock> const &) const = 0;

    point_type const &GetOrigin() const;
    point_type const &GetDx() const;

    static bool RegisterCreator(std::string const &k, std::function<Chart *()> const &);
    static Chart *Create(std::shared_ptr<data::DataTable> const &);

    template <typename U>
    static bool RegisterCreator(std::string const &k) {
        return RegisterCreator(k, [&]() -> Chart * { return new U; });
    }

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}  // namespace engine{
}  // namespace simpla{
#endif  // SIMPLA_CHART_H
