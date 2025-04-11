#pragma once

#include "Camera.h"

class OrbitalCamera : public Camera
{

    float xCenter = 0,
        yCenter = 0,
        zCenter = 0;

    float OrbitYaw = 0,
        OrbitPitch = 0;

    float radius = 5;

    float speed = 1,
        sensitivity = 1;

    virtual void HandleInput(const CameraControls& camControls, const MouseControls& mouseControls) override;
    virtual CameraType GetType() override { return CameraType::OrbitalCamera; }
    void RotateOrbit(float deltaPitch, float deltaYaw);
    void MoveCamera();
};