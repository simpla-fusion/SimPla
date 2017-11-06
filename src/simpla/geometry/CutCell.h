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
struct Shape;
struct IntersectionCurveSurface;
struct CutCell {
   private:
    typedef CutCell this_type;

   public:
   protected:
    CutCell();
    template <typename... Args>
    explicit CutCell(Args &&... args) : CutCell() {
        SetUp(std::forward<Args>(args)...);
    };

   public:
    ~CutCell();

    template <typename... Args>
    static std::shared_ptr<this_type> New(Args &&... args) {
        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...));
    }

    void SetUp(std::shared_ptr<const Chart> const &, std::shared_ptr<const Shape> const &, Real tolerance);
    void TearDown();
    void TagCell(Array<unsigned int> *vertex_tags, Array<Real> *edge_tags, unsigned int tag = 0b001) const;

   protected:
    struct pimpl_s;
    pimpl_s *m_pimpl_ = nullptr;
};
}  //    namespace geometry{
}  // namespace simpla{
#endif  // SIMPLA_CUTCELL_H
