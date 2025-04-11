#include "CameraController.h"

void FreeRoamCameraController::HandleInput(const CameraControls& camControls, const MouseControls& mouseControls)
{
	float moveForward = 0,
		moveRight = 0,
		moveUp = 0;

	if (camControls.forward)
		moveForward += 0.1f;
	if (camControls.backward)
		moveForward -= 0.1f;
	if (camControls.left)
		moveRight -= 0.1f;
	if (camControls.right)
		moveRight += 0.1f;
	if (camControls.up)
		moveUp += 0.1f;
	if (camControls.down)
		moveUp -= 0.1f;


	if (mouseControls.pressed)
	{
		float yaw = -0.005f * (mouseControls.relativeX),
			pitch = -0.005f * (mouseControls.relativeY);

		cam->rotate(pitch, yaw);
	}

	//move camera
	cam->x += cam->forward[0] * moveForward + cam->right[0] * moveRight;
	cam->y += cam->forward[1] * moveForward + cam->right[1] * moveRight + moveUp;
	cam->z += cam->forward[2] * moveForward + cam->right[2] * moveRight;
}

void OrbitalCameraController::HandleInput(const CameraControls& camControls, const MouseControls& mouseControls)
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


	if (mouseControls.pressed)
	{
		deltaYaw += -0.5f * (mouseControls.relativeX),
	    deltaPitch += -0.5f * (mouseControls.relativeY);
	}

	Rotate(deltaPitch, deltaYaw);
	radius += deltaRadius;

	MoveCamera();
}

void OrbitalCameraController::Rotate(float deltaPitch, float deltaYaw)
{
	OrbitYaw += deltaYaw;
	OrbitPitch += deltaPitch;

	// Clamp pitch to avoid gimbal lock
	OrbitPitch = std::fmax(-89.0f,
		std::fmin(89.0f, OrbitPitch));
	OrbitYaw = fmodf(OrbitYaw, 360.0f);
}

void OrbitalCameraController::MoveCamera()
{
	float pitchRad = OrbitPitch * 3.14f / 180.0f,
		yawRad = OrbitYaw * 3.14f / 180.0f;

	cam->x = radius * cosf(pitchRad) * sinf(yawRad);
	cam->y = radius * sinf(pitchRad);
	cam->z = radius * cosf(pitchRad) * cosf(yawRad);

	cam->setRotation(-pitchRad, yawRad);
}