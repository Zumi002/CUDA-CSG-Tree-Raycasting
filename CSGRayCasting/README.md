# CSG Tree Raycaster (CUDA)

CSG Tree Raycaster is a **Constructive Solid Geometry (CSG) Raycaster** implemented in **CUDA**.  
It uses **SDL2** with **OpenGL** for visualization and supports real-time interaction.  
The algorithm is based on **Andrew Kensler’s** method, described in  
[Ray Tracing CSG Objects Using Single Hit Intersections](https://xrt.wdfiles.com/local--files/doc%3Acsg/CSG.pdf),  
along with insights from  
[Spatially Efficient Tree Layout for GPU Ray-tracing of Constructive Solid Geometry Scenes](https://ceur-ws.org/Vol-1576/090.pdf).

###### Example: cylinder and sphere
<p align="center">
  <img src="Images/AppImage1.png"/>
</p>

## Features

### CUDA-Accelerated Raycasting
- Fast **GPU-based** ray tracing of CSG scenes.
- Efficient **tree traversal and intersection handling**.
- Supports **Phong lighting model** for shading.

### Custom CSG File Format
- Scene definition is **node-based** using **preorder traversal**.
- Primitives: **Sphere, Cylinder, Cube**.
- Operators: **Union, Intersection, Difference**.

### Interactive Controls
- **Move camera**: `W/S` (forward/backward), `A/D` (left/right), `Space/LShift` (up/down).
- **Rotate camera**: Hold **left mouse button** and move mouse.
- **Adjust lighting**: Change light direction dynamically.

###### Example: cube cut edges
<p align="center">
  <img src="Images/AppImage2.png"/>
</p>

## CSG Scene File Format

CSG trees are stored as text files where each node is defined in **preorder**.  
Primitives require additional parameters:

#### Node Types:
- **Operations**: `Union, Intersection, Difference`
- **Primitives**:
  - **Sphere**: `(float)posX (float)posY (float)posZ (hex)color (float)radius`
  - **Cylinder**: `(float)posX (float)posY (float)posZ (hex)color (float)radius (float)height (float)rotX (float)rotY (float)rotZ`
  - **Cube**: `(float)posX (float)posY (float)posZ (hex)color (float)edgeLength`

#### Example Scene: wikipedia example
```
Difference
	Intersection
		Cube 0 0 0 FF0000 2
		Sphere 0 0 0 0000FF 1.35
	Union
		Union
			Cylinder 0 0 0 00FF00 0.7 2 90 0 0
			Cylinder 0 0 0 00FF00 0.7 2 0 0 0
		Cylinder 0 0 0 00FF00 0.7 2 0 0 90
```
For clarity, **tabulation is recommended** to separate tree levels.

#### It represents:
<p align="center">
  <img src="Images/Csg_tree.png"/>
</p>
#### And it should look like:
<p align="center">
  <img src="Images/AppImage3.png"/>
</p>

#### Other test scences
Other test scenes available in the **Test** folder.

## Acknowledgments
This project is inspired by research in CSG ray tracing, particularly:
- **Andrew Kensler** - [Ray Tracing CSG Objects Using Single Hit Intersections](https://xrt.wdfiles.com/local--files/doc%3Acsg/CSG.pdf)
- **D.Y. Ulyanov**, **D.K. Bogolepov**, **V.E. Turlapov** - [Spatially Efficient Tree Layout for GPU Ray-tracing of Constructive Solid Geometry Scenes](https://ceur-ws.org/Vol-1576/090.pdf)

###### Example: shpere cut by cubes and cylinder
<p align="center">
  <img src="Images/AppImage4.png"/>
</p>

###### Example: Cheese512 (Cube with 512 spheres cutout)
<p align="center">
  <img src="Images/AppImage5.png"/>
</p>

