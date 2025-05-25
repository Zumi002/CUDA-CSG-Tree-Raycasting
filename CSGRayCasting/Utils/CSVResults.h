#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <exception>

struct BenchmarkResults
{
    std::string treeName;
    float FPS[3][2]; // fps (alg1, alg2, alg3), (1% low, avgFPS)

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

        void SaveResult(BenchmarkResults result, int alg)
        {
            std::ofstream file(filePath);
            file << "TreeName,1%low_SingleHit,AvgFPS_SingleHit,"
                "1%low_ClassicRaycast,AvgFPS_ClassicRaycast,"
                "1%low_Raymarch,AvgFPS_Raymarch\n";

            bool updated = false;
            file << std::fixed << std::setprecision(2);
            for (auto& r : results) {
                
                if (!updated && r.treeName == result.treeName)
                {
                    r.FPS[alg][0] = result.FPS[alg][0];
                    r.FPS[alg][1] = result.FPS[alg][1];
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
                ss >> result.FPS[2][1]; 

                results.push_back(result);
            }
        }

};