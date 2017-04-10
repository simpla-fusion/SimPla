//
// Created by salmon on 17-3-19.
//

#ifndef SIMPLA_ENGINE_ALL_H
#define SIMPLA_ENGINE_ALL_H

#include "Atlas.h"
#include "Attribute.h"
#include "Context.h"
#include "Domain.h"

#include "Mesh.h"
#include "Model.h"
#include "Patch.h"
#include "Task.h"
#include "TimeIntegrator.h"
/**
 *
 * @startuml
 *
 * note right of Atlas
 *    <latex>\mathcal{A}\equiv\left\{ \mathcal{O}_{\alpha},\varphi_{\alpha}\right\} </latex>
 *      <latex>X=\bigcup_{\alpha} \mathcal{O}_{\alpha} </latex>
 *  The set <latex> \mathcal{O} </latex> is known as <b>coordinate patch </b>,
 * If <latex> \mathcal{O} = \bar{\varphi} ^{-1} \left[ \mathbb{Z} \right]^n  </latex>, it is called as  <b>Mesh Block</b>.
 * end note
 *
 * class GeoObject{
 * }
 * class Atlas{
  *      Patch * Pop(index_box_type,int level)
 *      void Push(Patch *)
 *      id_type AddBlock(index_box_type,int level)
 *      index_box_type  GetBlock(id_type)
 * }
 * Atlas o-- "0..*" Patch
 * Atlas o-- "0..*" MeshBlock
 *
 * class Domain{
 *      GeoObject * m_geo_object_
 *      Chart * m_chart_
 *      Worker * m_worker_
 *      int CheckOverlap(MeshBlock)
 *      void Push(Patch)
 *      Patch Pop()
 *      Worker & GetWorker()
 * }
 * Domain *-- GeoObject
 * Domain *-- Worker
 * Domain *-- "1" Chart
 *
 * class IdRange{
 * }
 *
 * class Worker{
 *      void SetMesh(Mesh*)
 *      {abstract} void Register(AttributeGroup *);
 *      {abstract} void Deregister(AttributeGroup *);
 *      {abstract} void Push(Patch const &);
 *      {abstract} void Pop(Patch *) const;
 * }
 *
 * class ConcreteWorker<Mesh>{
 *      void Register(AttributeGroup*)
 *      void Initialize(Real time_now)
 *      void Run(Real time_now,Real dt)
 * }
 * ConcreteWorker -up-|> Worker
 * ConcreteWorker o-- Field
 *
 *
 * class Chart{
 *      point_type origin;
 *      point_type dx;
 *      Chart* Coarsen(int)const
 *      Chart* Refine(int)const
 *      Chart* Shift(point_type)const
 *      Chart* Rotate(Real a[])const
 *      {abstract} Mesh * CreateView(index_box_type)const
 * }
 * note bottom of Chart
 *   <b>Chart/Local Coordinates</b>
 *   A homeomorphism
 *     <latex>\varphi:\mathcal{O}\rightarrow\mathbb{R}^{n}\left[x^{0},...,x^{n-1}\right] </latex>
 *     <latex>\bar{\varphi}:\mathcal{O}\rightarrow\mathbb{Z}^{n}\left[x^{0},...,x^{n-1}\right] </latex>
 *    is called a <b>chart</b> or alternatively <b>local coordinates</b>.
 *    Each point <latex> x\in\mathcal{O} </latex> is then uniquely associated with
 *    an n-tuple of real numbers - its coordinates.
 *    The boundary of Chart is not defined.
 * end note
 * Chart *-- "0..*" MeshBlock
 * class MeshBlock{
 * }
 * MeshBlock .. DataBlock
 *
 * abstract  Mesh {
 *      {abstract} GeoObject boundary()const
 *      Range range(iform)
 *      {abstract} void Register(AttributeGroup *);
 *      {abstract} void Deregister(AttributeGroup *);
 *      {abstract} void Push(Patch const &);
 *      {abstract} void Pop(Patch *) const;
 * }
 * note right of Mesh
 *    <latex>\left\{ \mathcal{O}_{\alpha},\varphi_{\alpha}\right\} </latex>
 * end note
 * Mesh *-- Chart
 * Mesh *-- GeoObject
 * Mesh *-- IdRange
 * Mesh <.. Domain :create
 * class Patch {
 * }
 * Patch *-- "1" MeshBlock
 * Patch *-- "0..*" DataBlock
 * class IdRange{
 * }
 *
 * class MeshView<TGeometry> {
 *      GeoObject boundary()const
 *      void Register(AttributeGroup*)
 *      point_type vertex(index_tuple)const
 *      std::pair<index_tuple,point_type> map(point_type const &)const
 * }
 *
 *
 *
 *
 * class Attribute {
 *      void Register(AttributeGroup*)
 *      void SetMesh(Mesh *);
 *      {abstract} int GetIFORM()const;
 *      {abstract} int GetDOF()const
 *      Push(DataBlock);
 *      DataBlock Pop();
 * }
 *
 * class AttributeGroup {
 *      void Register(AttributeGroup*);
 *      void Detach(Attribute *attr);
 *      void Attach(Attribute *attr);
 *      void Push(Patch );
 *      Patch Pop();
 * }
 * AttributeGroup o-- Attribute

 * class Field<Mesh>{
  *    int GetIFORM()const;
 *     int GetDOF()const
 * }
 *
 * Field -up-|> Attribute
 *
 *
 * MeshView -up-|> Mesh
 * Chart <|-- CartesianGeometry
 * Chart <|-- CylindricalGeometry
 * MeshView .. CylindricalGeometry
 * MeshView .. CartesianGeometry
 *
 * Patch <..> Domain : push/pop
 * DataBlock <..> Attribute : push/pop
 * @enduml
 * @startuml
 * DataBlock <..> Attribute : push/pop

 *
 *
 * @enduml
 *
 */

/**
 *
 * @startuml
 * actor Main
 * Main -> DomainView : Set U as MeshView
 * activate DomainView
 *     alt if MeshView=nullptr
 *          create MeshView
 *     DomainView -> MeshView : create U as MeshView
 *     MeshView --> DomainView: return MeshView
 *     end
 *     DomainView --> Main : Done
 * deactivate DomainView
 * @enduml
 * @startuml
 * actor Main
 * Main -> DomainView : Dispatch
 * activate DomainView
 *     DomainView->MeshView:  Dispatch
 *     MeshView->MeshView: SetMeshBlock
 *     activate MeshView
 *     deactivate MeshView
 *     MeshView -->DomainView:  Done
*      DomainView --> Main : Done
 * deactivate DomainView
 * @enduml
 * @startuml
 * Main ->DomainView: Update
 * activate DomainView
 *     DomainView -> AttributeView : Update
 *     activate AttributeView
 *          AttributeView -> Field : Update
 *          Field -> AttributeView : Update
 *          activate AttributeView
 *               AttributeView -> DomainView : get DataBlock at attr.id()
 *               DomainView --> AttributeView : return DataBlock at attr.id()
 *               AttributeView --> Field : return DataBlock is ready
 *          deactivate AttributeView
 *          alt if data_block.isNull()
 *              Field -> Field :  create DataBlock
 *              Field -> AttributeView : send DataBlock
 *              AttributeView --> Field : Done
 *          end
 *          Field --> AttributeView : Done
 *          AttributeView --> DomainView : Done
 *     deactivate AttributeView
 *     DomainView -> MeshView : Update
 *     activate MeshView
 *          alt if isFirstTime
 *              MeshView -> AttributeView : Set Initialize Value
 *              activate AttributeView
 *                   AttributeView --> MeshView : Done
 *              deactivate AttributeView
 *          end
 *          MeshView --> DomainView : Done
 *     deactivate MeshView
 *     DomainView -> Worker : Update
 *     activate Worker
 *          alt if isFirstTime
 *              Worker -> AttributeView : set initialize value
 *              activate AttributeView
 *                  AttributeView --> Worker : Done
 *              deactivate AttributeView
 *          end
 *          Worker --> DomainView : Done
 *     deactivate Worker
 *     DomainView --> Main : Done
 * deactivate DomainView
 * deactivate Main
 * @enduml
 */
#endif  // SIMPLA_ENGINE_ALL_H
