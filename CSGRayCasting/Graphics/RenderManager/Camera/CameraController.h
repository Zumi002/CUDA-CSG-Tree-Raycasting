#pragma once

#include "Camera.h"
#include "../../Controls/InputManager.h"

class CameraController
{
   public:
   Camera* cam;
   virtual void HandleInput(const CameraControls& camControls, const MouseControls& mouseControls) = 0;
};

class FreeRoamCameraController : public CameraController
{
   virtual void HandleInput(const CameraControls& camControls, const MouseControls& mouseControls) override;
};

class OrbitalCameraController : public CameraController
{
    float xCenter = 0,
        yCenter = 0,
        zCenter = 0;
    float OrbitYaw = 0,
        OrbitPitch = 0;
    float radius = 5;
    virtual void HandleInput(const CameraControls& camControls, const MouseControls& mouseControls) override;

    void Rotate(float deltaPitch, float deltaYaw);

    void MoveCamera();
};