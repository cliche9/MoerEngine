/**
 * Soft Raster OIT — Fine Rasterization (Write Mode)
 *
 * Writes fragments to pre-allocated positions in the fragment buffer.
 * pixel_frag_count_buf is reused as a per-pixel atomic write counter (cleared
 * to 0 before this pass).
 */
#define WRITE_MODE 1

#include "core/common/Bindless.hlsl"
#include "core/common/Common.hlsl"
BINDLESS_BINDINGS(3, 2, 4, 5)

#include "shared/raster/software_rasterizer_oit/SoftRasterOITData.h"
#include "shared/raster/software_rasterizer_oit/SoftRasterParam.h"
#include "shared/raster/aoit/AOITData.h"
#include "shared/scene/SharedSceneStruct.h"
#include "materials/Brdf.hlsli"
#include "pipelines/raster/deferred/lighting/Lighting.hlsli"

// UAV bindings (set 0) — write mode with Lucid-style bin lists
[[vk::binding(0, 0)]] RWBuffer<uint4> triangle_buf;
[[vk::binding(1, 0)]] RWBuffer<uint>  tile_count_buf;
[[vk::binding(2, 0)]] RWBuffer<uint>  tile_tri_buf;
[[vk::binding(3, 0)]] RWBuffer<uint>  tile_offset_buf;
[[vk::binding(4, 0)]] RWBuffer<uint>  bin_list_buf;
[[vk::binding(5, 0)]] RWBuffer<uint>  bin_count_buf;
[[vk::binding(6, 0)]] RWBuffer<uint>  pixel_frag_count_buf;
[[vk::binding(7, 0)]] RWBuffer<uint>  pixel_frag_offset_buf;
[[vk::binding(8, 0)]] RWBuffer<uint2> fragment_buf;
[[vk::binding(9, 0)]] RWBuffer<float4> fragment_shade_buf;
[[vk::binding(10, 0)]] RWBuffer<uint>  debug_stats_buf;

[[vk::push_constant]] ConstantBuffer<Moer::SoftRasterTileParam> param;

#include "pipelines/raster/software_rasterizer_oit/SoftRasterFineRasterCore.hlsli"
