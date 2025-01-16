#include "Camera.h"


void Camera::updateVectors()
{
    // end result of applying rotation matrix to vectors
    // forward = [0,0,-1]
    // right = [1,0,0]
    // up = [0,1,0]
    forward[0] = -sin(rotY) * cos(rotX);
    forward[1] = sin(rotX);
    forward[2] = -cos(rotY) * cos(rotX);

    normalizeVector(forward);

    right[0] = cos(rotY);
    right[1] = 0;
    right[2] = -sin(rotY);
    normalizeVector(right);

    //here just cross product
    up[0] = forward[1] * right[2] - forward[2] * right[1];
    up[1] = forward[2] * right[0] - forward[0] * right[2];
    up[2] = forward[0] * right[1] - forward[1] * right[0];

    normalizeVector(up);
}

void Camera::normalizeVector(float* vec)
{
    float length = sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
    if (length > 0) {
        vec[0] /= length;
        vec[1] /= length;
        vec[2] /= length;
    }
}