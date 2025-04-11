#include "FreeRoamCamera.h"

void FreeRoamCamera::HandleInput(const CameraControls& camControls, const MouseControls& mouseControls)
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

	moveUp *= speed;
	moveRight *= speed;
	moveForward *= speed;

	if (mouseControls.pressed)
	{
		float yaw = -0.005f * (mouseControls.relativeX) * sensitivity,
			pitch = -0.005f * (mouseControls.relativeY) * sensitivity;

		rotate(pitch, yaw);
	}

	//move camera
	x += forward[0] * moveForward + right[0] * moveRight;
	y += forward[1] * moveForward + right[1] * moveRight + moveUp;
	z += forward[2] * moveForward + right[2] * moveRight;
}