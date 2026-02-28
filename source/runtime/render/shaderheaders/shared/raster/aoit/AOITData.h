/**
 * AOIT - Per-Pixel Linked List Order-Independent Transparency
 *
 * Shared defines between HLSL and C++.
 *
 * Fragment pool entry layout (uint4 per slot):
 *   .x = next pointer (0xFFFFFFFF = end of list)
 *   .y = depth (asuint(float))
 *   .z = packed RG (f32tof16(R) | f32tof16(G) << 16)
 *   .w = packed BA (f32tof16(B) | f32tof16(A) << 16)
 *
 * Counter buffer layout (RWBuffer<uint>):
 *   [0] = atomic allocation counter
 *   [1] = max_fragments (pool capacity)
 *
 * Head pointer buffer (RWBuffer<uint>):
 *   Per-pixel head of linked list, 0xFFFFFFFF = empty.
 *
 * CPP:  #include "shaderheaders/shared/raster/aoit/AOITData.h"
 * HLSL: #include "shared/raster/aoit/AOITData.h"
 */
#pragma once

#define AOIT_INVALID_POINTER 0xFFFFFFFF
#define AOIT_MAX_SORT_COUNT  32
