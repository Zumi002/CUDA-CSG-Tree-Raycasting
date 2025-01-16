#pragma once

namespace CSG
{
	enum CSGActions
	{
		GotoLft = 1 << 1,
		GotoRgh = 1 << 2,
		Compute = 1 << 3,
		LoadLft = 1 << 4,
		LoadRgh = 1 << 5,
		SaveLft = 1 << 6
	};

	enum HitActions
	{
		MissAction = 1 << 1,
		RetL = 1 << 2,
		RetR = 1 << 3,
		LoopL = 1 << 4,
		LoopR = 1 << 5,
		LoopRIfCloser = 1 << 6,
		LoopLIfCloser = 1 << 7,
		RetLIfCloser = 1 << 8,
		RetRIfCloser = 1 << 9,
		FlipR = 1 << 10
	};

	enum CSGRayHit
	{
		Enter = 1<<1,
		Exit = 1<<2,
		Miss = 1<<3,
		Flip = 1<<4
	};


}