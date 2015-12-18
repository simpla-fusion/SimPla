/*=========================================================================

 Program:   Visualization Toolkit
 Module:    vtkAMRSimPlaReader.h

 Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
 All rights reserved.
 See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notice for more information.

 =========================================================================*/
// .NAME vtkAMRSimPlaReader.h -- Reader for SimPla AMR datasets.
//
// .SECTION Description
// A concrete instance of vtkAMRBaseReader that implements functionality
// for reading SimPla AMR datasets.

#ifndef VTKAMRENZOREADER_H_
#define VTKAMRENZOREADER_H_

#define VTKIOAMR_EXPORT __attribute__((visibility("default")))

/* AutoInit dependencies.  */
#include <vtkFiltersAMRModule.h>
#include <vtkAMRBaseReader.h>

#include <map>     // For STL map
#include <string>  // For std::string


#include <vtksys/SystemTools.hxx>
#include <bits/unique_ptr.h>

#include <vector> // for STL vector
#include <string> // for STL string
#include <cassert>       // for assert()

class vtkOverlappingAMR;

class vtkDataArray;

class vtkDataSet;

/*****************************************************************************
*
* Copyright (c) 2000 - 2009, Lawrence Livermore National Security, LLC
* Produced at the Lawrence Livermore National Laboratory
* LLNL-CODE-400124
* All rights reserved.
*
* This file was adapted from the VisIt SimPla reader (avtSimPlaFileFormat). For
* details, see https://visit.llnl.gov/.  The full copyright notice is contained
* in the file COPYRIGHT located at the root of the VisIt distribution or at
* http://www.llnl.gov/visit/copyright.html.
*
*****************************************************************************/

static std::string GetSimPlaDirectory(const char *path)
{
    return (vtksys::SystemTools::GetFilenamePath(std::string(path)));
}


class VTKIOAMR_EXPORT vtkAMRSimPlaReader : public vtkAMRBaseReader
{
public:
    static vtkAMRSimPlaReader *New();

    vtkTypeMacro(vtkAMRSimPlaReader, vtkAMRBaseReader);

    void PrintSelf(ostream &os, vtkIndent indent);

    // Description:
    // Set/Get whether data should be converted to CGS
    vtkSetMacro(ConvertToCGS, int);

    vtkGetMacro(ConvertToCGS, int);

    vtkBooleanMacro(ConvertToCGS, int);

    // Description:
    // See vtkAMRBaseReader::GetNumberOfBlocks
    int GetNumberOfBlocks();

    // Description:
    // See vtkAMRBaseReader::GetNumberOfLevels
    int GetNumberOfLevels();

    // Description:
    // See vtkAMRBaseReader::SetFileName
    void SetFileName(const char *fileName);

protected:
    vtkAMRSimPlaReader();

    ~vtkAMRSimPlaReader();

    // Description:
    // Parses the parameters file and extracts the
    // conversion factors that are used to convert
    // to CGS units.
    void ParseConversionFactors();

    // Description:
    // Given an array name of the form "array[idx]" this method
    // extracts and returns the corresponding index idx.
    int GetIndexFromArrayName(std::string arrayName);

    // Description:
    // Given the label string, this method parses the Attribute label and
    // the string index.
    void ParseLabel(const std::string labelString, int &idx, std::string &label);

    // Description:
    // Given the label string, this method parses the corresponding Attribute
    // index and conversion factor
    void ParseCFactor(const std::string labelString, int &idx, double &factor);

    // Description:
    // Given the variable name, return the conversion factor used to convert
    // the data to CGS. These conversion factors are read directly from the
    // parameters file when the filename is set.
    double GetConversionFactor(const std::string name);

    // Description:
    // See vtkAMRBaseReader::ReadMetaData
    void ReadMetaData();

    // Description:
    // See vtkAMRBaseReader::GetBlockLevel
    int GetBlockLevel(const int blockIdx);

    void ComputeStats(std::vector<int> &blocksPerLevel, double min[3]);

    // Description:
    // See vtkAMRBaseReader::FillMetaData
    int FillMetaData();

    // Description:
    // See vtkAMRBaseReader::GetAMRGrid
    vtkUniformGrid *GetAMRGrid(const int blockIdx);

    // Description:
    // See vtkAMRBaseReader::GetAMRGridData
    void GetAMRGridData(
            const int blockIdx, vtkUniformGrid *block, const char *field);

    // Description:
    // See vtkAMRBaseReader::GetAMRGridData
    void GetAMRGridPointData(
            const int vtkNotUsed(blockIdx), vtkUniformGrid *vtkNotUsed(block), const char *vtkNotUsed(field)) { ; };

    // Description:
    // See vtkAMRBaseReader::SetUpDataArraySelections
    void SetUpDataArraySelections();

    int ConvertToCGS;
    bool IsReady;

private:
    vtkAMRSimPlaReader(const vtkAMRSimPlaReader &) = delete; // Not Implemented
    void operator=(const vtkAMRSimPlaReader &) = delete; // Not Implemented

    struct pimpl_s;

    std::unique_ptr<pimpl_s> m_pimpl_;


    std::map<std::string, int> label2idx;
    std::map<int, double> conversionFactors;
};

#endif /* VTKAMRENZOREADER_H_ */
