#pragma once

#include "Camera.h"

class FreeRoamCamera : public Camera
{
    float speed = 1,
        sensitivity = 1;
    virtual void HandleInput(const CameraControls& camControls, const MouseControls& mouseControls) override;
    virtual CameraType GetType() override { return CameraType::FreeRoamCamera; }
};