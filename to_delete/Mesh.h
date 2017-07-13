//
// Created by salmon on 16-11-19.
//

#ifndef SIMPLA_MESHBASE_H
#define SIMPLA_MESHBASE_H

#include <simpla/data/data.h>
#include <simpla/engine/Attribute.h>
#include <simpla/geometry/Chart.h>
#include <simpla/geometry/GeoObject.h>
#include <simpla/engine/SPObject.h>
namespace simpla {
namespace mesh {

// typedef EntityId64 EntityId;

#define MAX_POLYGON 20

enum MeshEntityType {
    //    VERTEX = 0, EDGE = 1, FACE = 2, VOLUME = 3,
    //    FIBER = 10 // points in cell

    TRIANGLE = (3 << 2) | 2,

    QUADRILATERAL = (4 << 2) | 2,

    // place Holder

    POLYGON = ((-1) << 2) | 2,

    // custom polygon

    TETRAHEDRON = (6 << 2) | 3,
    PYRAMID,
    PRISM,
    KNIFE,

    HEXAHEDRON = MAX_POLYGON + 12,
    // place Holder
    POLYHEDRON = MAX_POLYGON + (1 << 5),
    // custom POLYHEDRON

    MAX_POLYHEDRON = MAX_POLYGON + (1 << 6)

};
/**
 *   |<-----------------------------     valid   --------------------------------->|
 *   |<- not owned  ->|<-------------------       owned     ---------------------->|
 *   |----------------*----------------*---*---------------------------------------|
 *   |<---- ghost --->|                |   |                                       |
 *   |<------------ shared  ---------->|<--+--------  not shared  ---------------->|
 *   |<------------- DMZ    -------------->|<----------   not DMZ   -------------->|
 *
 */

enum MeshZoneTag {
    SP_ES_NULL = 0x00,        //                          0b000000
    SP_ES_ALL = 0x0F,         //                            0b001111 SP_ES_NOT_SHARED| SP_ES_SHARED |
                              //                            SP_ES_OWNED | SP_ES_NOT_OWNED
    SP_ES_OWNED = 0x01,       //                            0b000001 owned by local get_mesh block
    SP_ES_NOT_OWNED = 0x02,   //                        0b000010 not owned by local get_mesh block
    SP_ES_SHARED = 0x04,      //                           0b000100 shared by two or more get_mesh grid_dims
    SP_ES_NOT_SHARED = 0x08,  //                       0b001000 not shared by other get_mesh grid_dims
    SP_ES_LOCAL = SP_ES_NOT_SHARED | SP_ES_OWNED,  //              0b001001
    SP_ES_GHOST = SP_ES_SHARED | SP_ES_NOT_OWNED,  //              0b000110
    SP_ES_NON_LOCAL = SP_ES_SHARED | SP_ES_OWNED,  //              0b000101
    SP_ES_INTERFACE = 0x010,                       //                        0b010000 interface(boundary) shared by two
                                                   //                        get_mesh grid_dims,
    SP_ES_DMZ = 0x100,
    SP_ES_NOT_DMZ = 0x200,
    SP_ES_VALID = 0x400,
    SP_ES_UNDEFINED = 0xFFFF
};

/**
 *  @ingroup diff_geo
 *  @addtogroup  mesh get_mesh
 *  @{
 *  Mesh<>
 *  Concept:
 *  - Mesh<> know local information of topology and vertex coordinates, and
 *  - only explicitly store vertex adjacencies;
 *  - Mesh<> do not know global coordinates, topology_dims;
 *  - Mesh<> do not know metric;
 *
 *  ## Summary
 *
 *
 * Cell shapes supported in '''libmesh''' http://libmesh.github.io/doxygen/index.html
 * - 3 and 6 nodes triangles (Tri3, Tri6)
 * - 4, 8, and 9 nodes quadrilaterals (Quad4, Quad8, Quad9)
 * - 4 and 6 nodes infinite quadrilaterals (InfQuad4, InfQuad6)
 * - 4 and 10 nodes tetrahedrals (Tet4, Tet10)
 * - 8, 20, and 27 nodes  hexahedrals (Hex8, Hex20, Hex27)
 * - 6, 15, and 18 nodes prisms (Prism6, Prism15, Prism18)
 * - 5 nodes  pyramids (Pyramid5)
 * - 8, 16, and 18 nodes  infinite hexahedrals (InfHex8, InfHex16, InfHex18) ??
 * - 6 and 12 nodes  infinite prisms (InfPrism6, InfPrism12) ??
 *
 *
 *
 *
 *   @} */

/**
 *  Define:
 *   A bundle is a triple \f$(E, p, B)\f$ where \f$E\f$, \f$B\f$ are sets and \f$p:E \rightarrow B\f$ a map
 *   - \f$E\f$ is called the total space
 *   - \f$B\f$ is the base space of the bundle
 *   - \f$p\f$ is the projection
 *
 */
class MeshBase {
   public:
    SP_OBJECT_BASE(MeshBase)
    SP_DEFAULT_CONSTRUCT(MeshBase);

    explicit MeshBase(std::shared_ptr<geometry::Chart> const &c = nullptr, std::string const &s_name = "");

    virtual ~MeshBase();
    virtual bool empty() const { return false; }
    virtual std::shared_ptr<data::DataTable> Serialize() const;
    virtual void Deserialize(const std::shared_ptr<data::DataTable> &t);
    virtual unsigned int GetNDims() const { return 3; }

    void SetChart(std::shared_ptr<geometry::Chart> const &);
    std::shared_ptr<geometry::Chart> GetChart() const;

    point_type const &GetCellWidth() const;
    point_type const &GetOrigin() const;

    void FitBoundBox(box_type const &);

    index_tuple GetIndexOffset() const;
    void SetDimensions(index_tuple const &);
    index_tuple GetDimensions() const;

    void SetPeriodicDimension(size_tuple const &x);
    size_tuple const &GetPeriodicDimension() const;

    void SetDefaultGhostWidth(index_tuple const &);
    index_tuple GetDefaultGhostWidth() const;
    index_tuple GetGhostWidth(int tag = VERTEX) const;

    void SetBlock(const engine::MeshBlock &);
    const engine::MeshBlock &GetBlock() const;
    id_type GetBlockId() const;

    box_type GetBox() const;

    virtual void DoUpdate() {}

    virtual index_box_type GetIndexBox(int tag) const = 0;

    virtual point_type local_coordinates(EntityId s, Real const *r) const = 0;

    virtual point_type map(point_type const &) const;

    point_type local_coordinates(EntityId s, point_type const &r) const { return local_coordinates(s, &r[0]); };

    template <typename... Args>
    point_type point(Args &&... args) const {
        return local_coordinates(std::forward<Args>(args)...);
    }
    template <typename... Args>
    point_type global_coordinates(Args &&... args) const {
        return map(local_coordinates(std::forward<Args>(args)...));
    }

   protected:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}  // namespace mesh
}  // namespace simpla

#endif  // SIMPLA_MESHBASE_H
