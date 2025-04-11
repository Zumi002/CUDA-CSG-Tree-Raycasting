#pragma once

#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <string>
#include <sstream>
#include "../../Controls/InputManager.h"

class Camera 
{
    
    void updateVectors();
    void normalizeVector(float* vec);
public:
    enum CameraType
    {
        BasicCamera,
        FreeRoamCamera,
        OrbitalCamera
    };
    float x, y, z;        // Position
    float rotX, rotY;     // Pitch and Yaw in radians
    float fov;            // Field of view in radians
    float speed, sensitivity;

    // Cached view vectors
    float forward[3];
    float right[3];
    float up[3];

	bool overwriteFile = false; // Flag to overwrite the file with camera settings

    Camera() : x(0), y(0), z(5), rotX(0), rotY(0), fov(90.0f * 3.14159f / 180.0f), speed(1), sensitivity(1) {
        updateVectors();
    }

    void setPosition(float posX, float posY, float posZ) {
        x = posX;
        y = posY;
        z = posZ;
    }

    void setRotation(float pitch, float yaw) {

        // Clamp pitch to avoid gimbal lock
        rotX = std::fmax(-89.0f * 3.14159f / 180.0f,
            std::fmin(89.0f * 3.14159f / 180.0f, pitch));
        rotY = yaw;
        updateVectors();
    }

    void rotate(float deltaPitch, float deltaYaw) 
    {
        setRotation(rotX + deltaPitch, rotY + deltaYaw);
    }

    void setFOV(float degrees) 
    {
        fov = degrees * 3.14159f / 180.0f;
    }

    void LoadCameraSetting(const std::string& file_name);
	void SaveCameraSetting(const std::string& file_name) const;
    virtual void HandleInput(const CameraControls& camControls, const MouseControls& mouseControls) {};
    virtual CameraType GetType() { return CameraType::BasicCamera; }
};

