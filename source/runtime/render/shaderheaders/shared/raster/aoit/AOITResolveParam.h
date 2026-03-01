/**
 * AOIT Resolve Pass - Push Constant Parameters
 *
 * Contains all bindless handles and uniforms required by the compute resolve
 * shader to reconstruct geometry from visibility data and evaluate shading.
 *
 * CPP:  #include "shaderheaders/shared/raster/aoit/AOITResolveParam.h"
 * HLSL: #include "shared/raster/aoit/AOITResolveParam.h"
 */
#pragma once

#ifdef __cplusplus
#include "misc/Traits.h"
namespace Moer::Render {
#else
namespace Moer {
#endif

struct AOITResolveParam {
    // Inverse view-projection matrix for screen-to-world reconstruction
    float4x4 clip2world;

    // Scene geometry buffer handles (for vertex attribute reconstruction)
    uint instance_buf_hdl;
    uint primitive_buf_hdl;
    uint index_buf_hdl;
    uint position_buf_hdl;

    uint packed_normal_buf_hdl;
    uint packed_tangent_buf_hdl;
    uint texcoord0_buf_hdl;
    uint material_buf_hdl;

    // Lighting
    uint light_buf_hdl;
    uint global_param_handle;

    // Extra ambient
    uint enable_extra_ambient;
    uint _pad0; // explicit padding: float3 must start on 16-byte boundary (DX cbuffer layout)

    float3 extra_ambient_color;
    float  extra_ambient_intensity;
};

#ifdef __cplusplus
}
#else
}
#endif
