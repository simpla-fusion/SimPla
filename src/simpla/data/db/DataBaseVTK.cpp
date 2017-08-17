//
// Created by salmon on 17-8-14.
//
#include "DataBaseVTK.h"
//#include <vtkDoubleArray.h>
//#include <vtkPoints.h>
//#include <vtkUnstructuredGrid.h>
//#include <vtkXMLUnstructuredGridWriter.h>
//#include "simpla/particle/ParticleData.h"

namespace simpla {
namespace data {
struct DataBaseVTK::pimpl_s {};

DataBaseVTK::DataBaseVTK() : m_pimpl_(new pimpl_s) {}
DataBaseVTK::~DataBaseVTK() { delete m_pimpl_; }

int DataBaseVTK::Connect(std::string const& authority, std::string const& path, std::string const& query,
                         std::string const& fragment) {
    return SP_FAILED;
}
bool DataBaseVTK::isNull() const { return false; }

int DataBaseVTK::Disconnect() { return SP_FAILED; }

int DataBaseVTK::Flush() { return SP_FAILED; }

std::shared_ptr<DataEntity> DataBaseVTK::Get(std::string const& URI) const { return nullptr; }
int DataBaseVTK::Set(std::string const& URI, const std::shared_ptr<DataEntity>& d) { return SP_FAILED; }
int DataBaseVTK::Add(std::string const& URI, const std::shared_ptr<DataEntity>& d) { return SP_FAILED; }
int DataBaseVTK::Delete(std::string const& URI) { return 0; }
int DataBaseVTK::Foreach(std::function<int(std::string const&, std::shared_ptr<DataEntity>)> const& f) const {
    return 0;
}
//
// void addParticlesToVTKDataSet(const ParticleData* particles, vtkPoints* pts, vtkUnstructuredGrid* dataSet);
// void writeParticles(double time, const ParticleData* particles, std::ostringstream& fileName) {
//    // Create a writer
//    auto writer = vtkXMLUnstructuredGridWriter::New();
//    // Get the filename with extension
//    fileName << "." << writer->GetDefaultFileExtension();
//    writer->SetFileName((fileName.str()).c_str());
//    // Create a pointer to a VTK Unstructured Grid data set
//    auto dataSet = vtkUnstructuredGrid::New();
//    // Set up pointer to point data
//    auto pts = vtkPoints::New();
//    // Count the total number of points
//    int num_pts = static_cast<int>(particles->size());
//    pts->SetNumberOfPoints(num_pts);
//    // Add the time
//    addTimeToVTKDataSet(time, dataSet);
//    // Add the particle data to the unstructured grid
//    addParticlesToVTKDataSet(particles, pts, dataSet);
//    // Set the points
//    dataSet->SetPoints(pts);
//    // Remove unused memory
//    dataSet->Squeeze();
//    // Write the data
//    writer->SetInput(dataSet);
//    writer->SetDataModeToBinary();
//    writer->Write();
//}
//
// void addParticlesToVTKDataSet(const ParticleData* particles, vtkPoints* pts, vtkUnstructuredGrid* dataSet) {
//    // Set up pointers for material property data
//    auto ID = vtkDoubleArray::New();
//    ID->SetNumberOfComponents(1);
//    ID->SetNumberOfTuples(pts->GetNumberOfPoints());
//    ID->SetName("ID");
//
//    auto radii = vtkDoubleArray::New();
//    radii->SetNumberOfComponents(3);
//    radii->SetNumberOfTuples(pts->GetNumberOfPoints());
//    radii->SetName("Radius");
//
//    auto axis_a = vtkDoubleArray::New();
//    axis_a->SetNumberOfComponents(3);
//    axis_a->SetNumberOfTuples(pts->GetNumberOfPoints());
//    axis_a->SetName("Axis a");
//
//    auto axis_b = vtkDoubleArray::New();
//    axis_b->SetNumberOfComponents(3);
//    axis_b->SetNumberOfTuples(pts->GetNumberOfPoints());
//    axis_b->SetName("Axis b");
//
//    auto axis_c = vtkDoubleArray::New();
//    axis_c->SetNumberOfComponents(3);
//    axis_c->SetNumberOfTuples(pts->GetNumberOfPoints());
//    axis_c->SetName("Axis c");
//
//    auto position = vtkDoubleArray::New();
//    position->SetNumberOfComponents(3);
//    position->SetNumberOfTuples(pts->GetNumberOfPoints());
//    position->SetName("Position");
//
//    auto velocity = vtkDoubleArray::New();
//    velocity->SetNumberOfComponents(3);
//    velocity->SetNumberOfTuples(pts->GetNumberOfPoints());
//    velocity->SetName("Velocity");
//
//    // Loop through particles
//    Vec vObj;
//    int id = 0;
//    double vec[3];
//    for (const auto& particle : *particles) {
//        // Position
//        vObj = particle->getCurrPos();
//        vec[0] = vObj.getX();
//        vec[1] = vObj.getY();
//        vec[2] = vObj.getZ();
//        pts->SetPoint(id, vec);
//
//        // ID
//        ID->InsertValue(id, particle->getId());
//
//        // Ellipsoid radii
//        vec[0] = particle->getA();
//        vec[1] = particle->getB();
//        vec[2] = particle->getC();
//        radii->InsertTuple(id, vec);
//
//        // Current direction A
//        vObj = particle->getCurrDirecA();
//        vec[0] = vObj.getX();
//        vec[1] = vObj.getY();
//        vec[2] = vObj.getZ();
//        axis_a->InsertTuple(id, vec);
//
//        // Current direction B
//        vObj = particle->getCurrDirecB();
//        vec[0] = vObj.getX();
//        vec[1] = vObj.getY();
//        vec[2] = vObj.getZ();
//        axis_b->InsertTuple(id, vec);
//
//        // Current direction C
//        vObj = particle->getCurrDirecC();
//        vec[0] = vObj.getX();
//        vec[1] = vObj.getY();
//        vec[2] = vObj.getZ();
//        axis_c->InsertTuple(id, vec);
//
//        // Velocity
//        vObj = particle->getCurrVeloc();
//        vec[0] = vObj.getX();
//        vec[1] = vObj.getY();
//        vec[2] = vObj.getZ();
//        velocity->InsertTuple(id, vec);
//
//        ++id;
//    }
//
//    // Add points to data set
//    dataSet->GetPointData()->AddArray(ID);
//    dataSet->GetPointData()->AddArray(radii);
//    dataSet->GetPointData()->AddArray(axis_a);
//    dataSet->GetPointData()->AddArray(axis_b);
//    dataSet->GetPointData()->AddArray(axis_c);
//    dataSet->GetPointData()->AddArray(velocity);
//}
}  // namespace data
}  // namespace simpla