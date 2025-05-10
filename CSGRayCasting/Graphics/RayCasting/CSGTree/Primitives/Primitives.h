#pragma once

#include <cuda_runtime.h>

#define M_DEGTORAD 0.017453292519943295769236907684886;

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

	void makeCube(float size)
	{
		cubeParameters.size = size;
	}

	void makeSphere(float radius)
	{
		sphereParameters.radius = radius;
	}

	void makeCylinder(float radius, float height, float rotX, float rotY, float rotZ)
	{
		cylinderParameters.radius = radius;
		cylinderParameters.height = height;

		rotX = rotX * M_DEGTORAD;
		rotY = rotY * M_DEGTORAD;
		rotZ = rotZ * M_DEGTORAD;

		double axisX = -sin(rotZ) * cos(rotY) + sin(rotY) * sin(rotX) * cos(rotZ),
			axisY = cos(rotX) * cos(rotZ),
			axisZ = sin(rotY) * sin(rotZ) + sin(rotX) * cos(rotY) * cos(rotZ);

		//final after  rotation vector [0, 1, 0] with y,x,z rotations
		cylinderParameters.axisX = -sin(rotZ) * cos(rotY) + sin(rotY) * sin(rotX) * cos(rotZ);
		cylinderParameters.axisY = cos(rotX) * cos(rotZ);
		cylinderParameters.axisZ = sin(rotY) * sin(rotZ) + sin(rotX) * cos(rotY) * cos(rotZ);

		double len = (axisX * axisX + axisY * axisY + axisZ * axisZ);

		axisX /= len;
		axisY /= len;
		axisZ /= len;

		cylinderParameters.axisX = axisX;
		cylinderParameters.axisY = axisY;
		cylinderParameters.axisZ = axisZ;
	}
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

		rotX = rotX * M_DEGTORAD;
		rotY = rotY * M_DEGTORAD;
		rotZ = rotZ * M_DEGTORAD;

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

struct  __align__(16)  CudaPrimitivePos
{
	float x;
	float y;
	float z;

	CudaPrimitivePos(float X, float Y, float Z)
	{
		x = X;
		y = Y;
		z = Z;
	}

	__device__ CudaPrimitivePos() {}
};

struct CudaPrimitiveColor
{
	float r;
	float g;
	float b;

	CudaPrimitiveColor(float R, float G, float B)
	{
		r = R;
		g = G;
		b = B;
	}
};

struct Primitives
{
	std::vector<CudaPrimitivePos> primitivePos;
	std::vector<CudaPrimitiveColor> primitiveColor;
	std::vector<Parameters> primitiveParameters;

	void addSphere(int Id, float X, float Y, float Z, float R, float G, float B, float Radius)
	{
		primitivePos.push_back(CudaPrimitivePos(X, Y, Z));
		primitiveColor.push_back(CudaPrimitiveColor(R, G, B));
		Parameters params = Parameters();
		params.makeSphere(Radius);
		primitiveParameters.push_back(params);
	}

	void addCylinder(int Id, float X, float Y, float Z, float R, float G, float B, float Radius, float Height, double RotX, double RotY, double RotZ)
	{
		primitivePos.push_back(CudaPrimitivePos(X, Y, Z));
		primitiveColor.push_back(CudaPrimitiveColor(R, G, B));
		Parameters params = Parameters();
		params.makeCylinder(Radius, Height, RotX, RotY, RotZ);
		primitiveParameters.push_back(params);
	}

	void addCube(int Id, float X, float Y, float Z, float R, float G, float B, float Size)
	{
		primitivePos.push_back(CudaPrimitivePos(X, Y, Z));
		primitiveColor.push_back(CudaPrimitiveColor(R, G, B));
		Parameters params = Parameters();
		params.makeCube(Size);
		primitiveParameters.push_back(params);
	}
};
