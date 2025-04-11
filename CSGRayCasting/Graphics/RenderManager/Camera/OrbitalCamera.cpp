#include "OrbitalCamera.h"

void OrbitalCamera::HandleInput(const CameraControls& camControls, const MouseControls& mouseControls)
{
	float deltaYaw = 0,
		deltaPitch = 0;
	float deltaRadius = 0;
	if (camControls.forward)
		deltaRadius -= 0.1f;
	if (camControls.backward)
		deltaRadius += 0.1f;
	if (camControls.left)
		deltaYaw -= 0.5f;
	if (camControls.right)
		deltaYaw += 0.5f;
	if (camControls.up)
		deltaPitch -= 0.5f;
	if (camControls.down)
		deltaPitch += 0.5f;

	deltaPitch *= speed;
	deltaYaw *= speed;

	if (mouseControls.pressed)
	{
		deltaYaw += -0.5f * (mouseControls.relativeX) * sensitivity,
			deltaPitch += -0.5f * (mouseControls.relativeY) * sensitivity;
	}

	RotateOrbit(deltaPitch, deltaYaw);
	radius += deltaRadius * speed;

	MoveCamera();
}

void OrbitalCamera::RotateOrbit(float deltaPitch, float deltaYaw)
{
	OrbitYaw += deltaYaw;
	OrbitPitch += deltaPitch;

	// Clamp pitch to avoid gimbal lock
	OrbitPitch = std::fmax(-89.0f,
		std::fmin(89.0f, OrbitPitch));
	OrbitYaw = fmodf(OrbitYaw, 360.0f);
}

void OrbitalCamera::MoveCamera()
{
	float pitchRad = OrbitPitch * 3.14f / 180.0f,
		yawRad = OrbitYaw * 3.14f / 180.0f;

	x = radius * cosf(pitchRad) * sinf(yawRad);
	y = radius * sinf(pitchRad);
	z = radius * cosf(pitchRad) * cosf(yawRad);

	setRotation(-pitchRad, yawRad);
}