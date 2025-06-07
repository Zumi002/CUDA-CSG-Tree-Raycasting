#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <exception>

#include "AdditionalStatistics.h"

struct BenchmarkResults
{
    std::string treeName;
    float FPS[3][2]; // fps (alg1, alg2, alg3), (1% low, avgFPS)
    float avgPrimitivesPerPixel;
    BenchmarkResults(){}

    BenchmarkResults(const std::string& treeName)
    {
        this->treeName = treeName;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                FPS[i][j] = 0;
            }
        }
        avgPrimitivesPerPixel = 0;
    }

};

class CSVResults
{
    std::vector<BenchmarkResults> results;
    std::string filePath;
    
    public:
        bool error = false;
        CSVResults(const std::string& filePath)
        {
            this->filePath = filePath;
            try
            {
                LoadFile();
            }
            catch (std::exception e)
            {
                fprintf(stderr, "There was an error parsing results\n");
                error = true;
            }
        }

        void SaveResult(BenchmarkResults result, int alg, bool collectsStats)
        {
            std::ofstream file(filePath);
            file << "TreeName,1%low_SingleHit,AvgFPS_SingleHit,"
                "1%low_ClassicRaycast,AvgFPS_ClassicRaycast,"
                "1%low_Raymarch,AvgFPS_Raymarch,AvgPrimitivesPerPixel\n";

            bool updated = false;
            file << std::fixed << std::setprecision(2);
            for (auto& r : results) 
            {
                
                if (!updated && r.treeName == result.treeName)
                {
                    r.FPS[alg][0] = result.FPS[alg][0];
                    r.FPS[alg][1] = result.FPS[alg][1];
                    if (collectsStats)
                    {
                        r.avgPrimitivesPerPixel = result.avgPrimitivesPerPixel;
                    }
                    updated = true;
                }
                file << r.treeName;
                for (int i = 0; i < 3; i++)
                {
                    for (int j = 0; j < 2; j++)
                    {
                        file << "," << r.FPS[i][j];
                    }
                }
                file << "," << r.avgPrimitivesPerPixel;
                file << "\n";
            }
            if (!updated)
            {
                file << result.treeName;
                for (int i = 0; i < 3; i++)
                {
                    for (int j = 0; j < 2; j++)
                    {
                        file << "," << result.FPS[i][j];
                    }
                }
                file << "," << result.avgPrimitivesPerPixel;
                file << "\n";
            }
        }

    private:
        void LoadFile()
        {
            results.clear();

            std::ifstream file(filePath);

            if (!file.is_open())
            {
                return;
            }

            std::string line;

            std::getline(file, line); //skip headers

            while (std::getline(file, line)) {
                std::stringstream ss(line);
                std::string field;
                BenchmarkResults result;

                std::getline(ss, result.treeName, ',');
                ss >> result.FPS[0][0]; ss.ignore();
                ss >> result.FPS[0][1]; ss.ignore();
                ss >> result.FPS[1][0]; ss.ignore();
                ss >> result.FPS[1][1]; ss.ignore();
                ss >> result.FPS[2][0]; ss.ignore();
                ss >> result.FPS[2][1]; ss.ignore();
                ss >> result.avgPrimitivesPerPixel;

                results.push_back(result);
            }
        }

};