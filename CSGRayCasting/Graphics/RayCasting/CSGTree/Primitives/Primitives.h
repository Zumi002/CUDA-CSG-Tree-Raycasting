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

	Primitive(int Id, float X, float Y, float Z, float R, float G, float B, float Radius, float height, double rotX, double rotY, double rotZ)
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

		rotX = rotX * 0.017453292519943295769236907684886;
		rotY = rotY * 0.017453292519943295769236907684886;
		rotZ = rotZ * 0.017453292519943295769236907684886;

		double axisX = -sin(rotZ) * cos(rotY) + sin(rotY) * sin(rotX) * cos(rotZ),
			axisY = cos(rotX) * cos(rotZ),
			axisZ = sin(rotY) * sin(rotZ) + sin(rotX) * cos(rotY) * cos(rotZ);

		//final after  rotation vector [0, 1, 0] with y,x,z rotations
		params.cylinderParameters.axisX = -sin(rotZ)*cos(rotY)+sin(rotY)*sin(rotX)*cos(rotZ);
		params.cylinderParameters.axisY = cos(rotX)*cos(rotZ);
		params.cylinderParameters.axisZ = sin(rotY)*sin(rotZ)+sin(rotX)*cos(rotY)*cos(rotZ);

		double len = (axisX*axisX+axisY*axisY+axisZ*axisZ);
		
		axisX /= len;
		axisY /= len;
		axisZ /= len;

		params.cylinderParameters.axisX = axisX;
		params.cylinderParameters.axisY = axisY;
		params.cylinderParameters.axisZ = axisZ;
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

	void addCylinder(int Id, float X, float Y, float Z, float R, float G, float B, float Radius, float Height, double RotX, double RotY, double RotZ)
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
