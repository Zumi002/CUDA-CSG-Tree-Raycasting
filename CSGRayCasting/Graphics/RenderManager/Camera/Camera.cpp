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

void Camera::LoadCameraSetting(const std::string& file_name)
{
	std::ifstream file(file_name);

	if (!file.is_open()) {
		std::cerr << "Failed to open camera.ini for reading!" << std::endl;
		return;
	}

	std::string line;
	while (std::getline(file, line)) {
		// Skip comments and section headers
		if (line.empty() || line[0] == '#' || line[0] == '[') continue;

		std::istringstream iss(line);
		std::string key;
		if (std::getline(iss, key, '=')) {
			std::string valueStr;
			if (std::getline(iss, valueStr)) {
				float value = std::stof(valueStr);

				if (key == "fov") fov = value;
				else if (key == "x") x = value;
				else if (key == "y") y = value;
				else if (key == "z") z = value;
				else if (key == "rotX") rotX = value;
				else if (key == "rotY") rotY = value;
				else if (key == "overwriteFile") {
					if (valueStr == "1") overwriteFile = true;
					else if (valueStr == "0") overwriteFile = false;
				}
				
			}
		}
	}
	updateVectors();

	file.close();

	std::cout << "Camera position loaded from "<<file_name << "\n";
}

void Camera::SaveCameraSetting(const std::string& file_name) const
{
	if (std::filesystem::exists(file_name) && !overwriteFile) {
		//std::cout << "File already exists and overwriteFile is set to false!" << std::endl;
		return;
	}
	std::ofstream file(file_name);

	if (!file.is_open()) {
		std::cerr << "Failed to open camera.ini for writing!" << std::endl;
		return;
	}

	file << std::fixed << std::setprecision(6); // consistent float formatting

	file << "[Camera]" << "\n";
	file << "fov=" << fov << "\n";
	file << "x=" << x << "\n";
	file << "y=" << y << "\n";
	file << "z=" << z << "\n";
	file << "rotX=" << rotX << "\n";
	file << "rotY=" << rotY << "\n";
	file << "overwriteFile=" << (overwriteFile ? "1" : "0") << "\n";

	file.close();

	std::cout << "Camera position saved to "<<file_name << "\n";
}


