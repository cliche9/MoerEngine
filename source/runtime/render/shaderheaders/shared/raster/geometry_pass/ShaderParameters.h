/**
 * 请统一Include如下文件，不要Include当前文件
 * CPP:
 *     #include "shaderheaders/shared/raster/ShaderParameters.h"
 * HLSL:
 *     #include "shared/raster/ShaderParameters.h"
 */
#pragma once

#ifdef CONST
#undef CONST
#endif

#ifdef __cplusplus
//#define CONST constexpr
#include "misc/Traits.h"
namespace Moer::Render {
#else
//#define CONST const
namespace Moer {
#endif

// MARK: Main Content Begin

struct GeometryPassBindlessParam {
    float4x4 world2clip;

    // Bindless handles for bindless rendering
    uint material_buf_hdl;  // Array<GMaterial>
    uint instance_buf_hdl;  // Array<GInstance>
    uint primitive_buf_hdl; // Array<GPrimitive>

    uint position_buf_hdl;       // Array<float3>
    uint packed_normal_buf_hdl;  // Array<uint> (packed normal)
    uint packed_tangent_buf_hdl; // Array<uint> (packed tangent)
    uint texcoord0_buf_hdl;      // Array<float2>

    // about material & alpha test
    uint  enable_alpha_test;
    float alpha_test_blend_pixel_cutoff;

    // forward-shading fields (used by TransparentBlendPass / AOIT collect)
    uint   light_buf_hdl;       // Array<GLight>
    uint   global_param_handle; // LightingData buffer handle
    uint   _pad0;               // explicit padding: float3 must start on 16-byte boundary (DX cbuffer layout)
    float3 extra_ambient_color;
    float  extra_ambient_intensity;
    uint   enable_extra_ambient;
    uint   _pad1;
};

// MARK: Main Content End

#ifdef __cplusplus
}
#else
}
#endif