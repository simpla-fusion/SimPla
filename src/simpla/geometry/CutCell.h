//
// Created by salmon on 17-7-26.
//

#ifndef SIMPLA_CUTCELL_H
#define SIMPLA_CUTCELL_H

#include <simpla/algebra/Array.h>
#include <simpla/utilities/SPDefines.h>
#include <memory>
namespace simpla {
namespace geometry {

class Chart;
struct GeoEntity;
struct CutCell {
   private:
    typedef CutCell this_type;

   protected:
    CutCell();
    explicit CutCell(std::shared_ptr<const GeoEntity> const &, std::shared_ptr<const Chart> const &c,
                     Real tolerance = SP_GEO_DEFAULT_TOLERANCE);

   public:
    ~CutCell();

    template <typename... Args>
    static std::shared_ptr<this_type> New(Args &&... args) {
        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...));
    }
    void TagCell(Array<unsigned int> *node_tags, Array<Real> *edge_tags, unsigned int tag = 0b001);
    void TagCell(Array<unsigned int> *node_tags, Array<Real> *edge_tags, unsigned int tag = 0b001) const;

    void SetChart(std::shared_ptr<Chart> const &c);
    std::shared_ptr<const Chart> GetChart() const;
    //    void SetShape(std::shared_ptr<GeoEntity> const &c) { m_shape_ = c; }
    //    std::shared_ptr<GeoEntity> GetShape() const { return m_shape_; }
    //    void SetTolerance(Real v) { m_tolerance_ = v; }
    //    Real GetTolerance() const { return m_tolerance_; }

   protected:
    struct pimpl_s;
    pimpl_s *m_pimpl_ = nullptr;
};
}  //    namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_CUTCELL_H
