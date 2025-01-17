#pragma once

struct SphereParameters
{
	float radius;
};

struct CylinderParameters
{
	float radius;
	float height;
	float axisX;
	float axisY;
	float axisZ;
};

struct CubeParameters
{
	float size;
};

union Parameters
{
	SphereParameters sphereParameters;
	CylinderParameters cylinderParameters;
	CubeParameters cubeParameters;
};

struct Primitive
{
	int id;
	float x;
	float y;
	float z;

	float r;
	float g;
	float b;

	Parameters params;

	Primitive(int Id, float X, float Y, float Z, float R, float G, float B, float Radius)
	{
		id = Id;
		x = X;
		y = Y;
		z = Z;
		r = R;
		g = G;
		b = B;
		params.sphereParameters.radius = Radius;
	}

	Primitive(int Id, float X, float Y, float Z, float R, float G, float B, float Radius, float height, float rotX, float rotY, float rotZ)
	{
		id = Id;
		x = X;
		y = Y;
		z = Z;
		r = R;
		g = G;
		b = B;
		params.cylinderParameters.radius = Radius;
		params.cylinderParameters.height = height;

		rotX = rotX * 3.14159f / 180.0f;
		rotY = rotY * 3.14159f / 180.0f;
		rotZ = rotZ * 3.14159f / 180.0f;

		//final after  rotation vector [0, 1, 0] with y,x,z rotations
		params.cylinderParameters.axisX = -sin(rotZ)*cos(rotY)+sin(rotY)*sin(rotX)*cos(rotZ);
		params.cylinderParameters.axisY = cos(rotX)*cos(rotZ);
		params.cylinderParameters.axisZ = sin(rotY)*sin(rotZ)+sin(rotX)*cos(rotY)*cos(rotZ);
	}

	Primitive(int Id, float X, float Y, float Z, float R, float G, float B)
	{
		id = Id;
		x = X;
		y = Y;
		z = Z;
		r = R;
		g = G;
		b = B;
	}

};

struct Primitives
{
	std::vector<Primitive> primitives;

	void addSphere(int Id, float X, float Y, float Z, float R, float G, float B, float Radius)
	{
		primitives.push_back(Primitive(Id, X, Y, Z, R, G, B, Radius));
	}

	void addCylinder(int Id, float X, float Y, float Z, float R, float G, float B, float Radius, float Height, float RotX, float RotY, float RotZ)
	{
		primitives.push_back(Primitive(Id, X, Y, Z, R, G, B, Radius, Height, RotX, RotY, RotZ));
	}

	void addCube(int Id, float X, float Y, float Z, float R, float G, float B, float Size)
	{
		Primitive cube = Primitive(Id, X, Y, Z, R, G, B);
		cube.params.cubeParameters.size = Size;
		primitives.push_back(cube);
	}
};
