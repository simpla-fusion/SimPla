//
// Created by salmon on 17-7-26.
//

#ifndef SIMPLA_CUTCELL_H
#define SIMPLA_CUTCELL_H

#include <simpla/algebra/Array.h>
#include <simpla/data/DataNode.h>
#include <simpla/utilities/Factory.h>
#include <simpla/utilities/SPDefines.h>
#include <memory>
namespace simpla {
namespace geometry {

class Chart;
class Surface;

struct CutCell : Factory<CutCell> {
   private:
    typedef CutCell this_type;

   public:
    static std::string FancyTypeName_s() { return "CutCell"; }
    static std::string RegisterName_s() { return "CutCell"; }
    virtual std::string FancyTypeName() const { return FancyTypeName_s(); }
    virtual std::string RegisterName() const { return RegisterName_s(); }

   protected:
    CutCell();
    CutCell(std::shared_ptr<const Chart> const &, std::shared_ptr<const Surface> const &, Real tolerance);

   public:
    ~CutCell() override;
    static std::shared_ptr<this_type> New(std::string const &key = "");
    static std::shared_ptr<this_type> New(std::shared_ptr<simpla::data::DataNode> const &d);
    template <typename... Args>
    static std::shared_ptr<this_type> New(Args &&... args) {
        auto res = New();
        //        res->SetUp(std::forward<Args>(args)...);
        return res;
    }
    virtual void SetUp(std::shared_ptr<const Chart> const &, std::shared_ptr<const Surface> const &, Real tolerance);

    virtual void Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg);
    virtual std::shared_ptr<simpla::data::DataNode> Serialize() const;

    virtual size_type IntersectAxe(index_tuple const &idx, int dir, index_type length, std::vector<Real> *u) const;
    void TagCell(Array<unsigned int> *vertex_tags, Array<Real> *edge_tags, unsigned int tag = 0b001) const;

   protected:
    std::shared_ptr<const Chart> m_chart_ = nullptr;
    std::shared_ptr<const Surface> m_surface_ = nullptr;
    Real m_tolerance_ = SP_GEO_DEFAULT_TOLERANCE;
};
}  //    namespace geometry{

}  // namespace simpla{
#endif  // SIMPLA_CUTCELL_H
