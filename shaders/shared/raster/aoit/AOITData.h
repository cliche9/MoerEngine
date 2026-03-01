/**
 * AOIT - Per-Pixel Linked List Order-Independent Transparency
 *         (Visibility Buffer variant)
 *
 * Shared defines between HLSL and C++.
 *
 * Fragment pool entry layout (uint4 per slot):
 *   .x = next pointer (0xFFFFFFFF = end of list)
 *   .y = asuint(depth)     — vis_depth upper 32 bits
 *   .z = packed_vis_info    — vis_depth lower 32 bits
 *   .w = reserved (0)
 *
 * 64-bit vis_depth = (uint64(asuint(depth)) << 32) | packed_vis_info
 *   Reversed-Z: larger depth = closer to camera.
 *   asuint(float) preserves ordering for positive floats,
 *   so InterlockedMax on vis_depth keeps the closest fragment.
 *
 * packed_vis_info (lower 32 bits) layout:
 *   bit  0       : is_front_face
 *   bits [1..19] : triangle_id  (19 bits, max 524287)
 *   bits [20..31]: instance_id  (12 bits, max 4095)
 *
 * Counter buffer layout (RWBuffer<uint>):
 *   [0] = atomic allocation counter
 *   [1] = max_fragments (pool capacity)
 *
 * Head pointer buffer (RWTexture2D<uint>):
 *   Per-pixel head of linked list, 0xFFFFFFFF = empty.
 *
 * CPP:  #include "shaderheaders/shared/raster/aoit/AOITData.h"
 * HLSL: #include "shared/raster/aoit/AOITData.h"
 */
#pragma once

#define AOIT_INVALID_POINTER 0xFFFFFFFF
#define AOIT_MAX_SORT_COUNT  32

// ---- 64-bit vis_depth packing constants ----
#define AOIT_FRONT_FACE_BIT 0u

#define AOIT_TRIANGLE_ID_BITS  19u
#define AOIT_TRIANGLE_ID_SHIFT 1u
#define AOIT_TRIANGLE_ID_MASK  0x7FFFFu // (1 << 19) - 1

#define AOIT_INSTANCE_ID_BITS  12u
#define AOIT_INSTANCE_ID_SHIFT 20u
#define AOIT_INSTANCE_ID_MASK  0xFFFu // (1 << 12) - 1
