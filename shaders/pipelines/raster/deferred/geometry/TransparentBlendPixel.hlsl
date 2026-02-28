#ifndef SHADOW_DEPTH_PASS
#define SHADOW_DEPTH_PASS 0
#endif

#include "core/common/Bindless.hlsl"
#include "core/common/Common.hlsl"
BINDLESS_BINDINGS(3, 2, 4, 5)
#include "shared/Geometry.h"
#include "shared/raster/ShaderParameters.h"
#include "shared/scene/SharedSceneStruct.h"

#include "materials/Material.hlsli"
#include "materials/Brdf.hlsli"
#include "pipelines/raster/deferred/lighting/Lighting.hlsli"
#include "pipelines/raster/deferred/geometry/GeometryPassCommon.hlsli"

[[vk::push_constant]] ConstantBuffer<Moer::GeometryPassBindlessParam> param;

float4 main(VsOutput input, bool is_front_face : SV_IsFrontFace) : SV_TARGET {
    ArrayBuffer material_buf = ArrayBuffer(param.material_buf_hdl);
    Moer::GMaterial mat = material_buf.Load<Moer::GMaterial>(input.material_id);

    if (mat.alpha_mode != Moer::EAlphaMode::Blend) {
        discard;
    }

    int albedo_map_hdl = asint(mat.albedo_map_hdl);
    float4 albedo_tex = GetTextureData<float4>(
        albedo_map_hdl,
        input.texcoord0,
        float4(1.0, 1.0, 1.0, 1.0),
        float4(1.0, 1.0, 1.0, 1.0)
    );

    float3 albedo_color = albedo_tex.rgb * mat.albedo_factor.rgb;
    float alpha = saturate(albedo_tex.a * mat.albedo_factor.a);

    if (param.enable_alpha_test != 0 && alpha < param.alpha_test_blend_pixel_cutoff) {
        discard;
    }

    float2 metallic_roughness = GetTextureData<float2>(
        asint(mat.metallic_roughness_map_hdl),
        input.texcoord0,
        float2(mat.metallic_factor, mat.roughness_factor),
        float2(mat.metallic_factor, mat.roughness_factor)
    );
    float metallic  = metallic_roughness.x;
    float roughness = metallic_roughness.y;

    float3 N = GetNormalFromNormalMap(asint(mat.normal_map_hdl), input.texcoord0, normalize(input.normal), normalize(input.tangent));
    N = is_front_face ? N : -N;

    ArrayBuffer global_params = ArrayBuffer(param.global_param_handle);
    Moer::LightingData lighting_data = global_params.Load<Moer::LightingData>(0);

    float3 V = normalize(lighting_data.camera_position - input.world_position.xyz);
    float NoV = saturate(dot(N, V));

    BRDFContext brdf_ctx;
    brdf_ctx.Init(
        roughness,
        albedo_color,
        metallic,
        N,
        V,
        TextureHandle(lighting_data.lut_ggx_emu_handle).Sample2D<float3>(float2(NoV, roughness)),
        TextureHandle(lighting_data.lut_ggx_eavg_handle).Sample2D<float3>(float2(0.0, roughness))
    );

    brdf_ctx.SetConfig(
        lighting_data.brdf_enable_multi_scatter,
        lighting_data.brdf_NDF_mode,
        lighting_data.brdf_G_mode,
        lighting_data.brdf_G_is_ibl
    );

    ArrayBuffer light_buf = ArrayBuffer(param.light_buf_hdl);

    LightContext light_ctx;
    light_ctx.Init(brdf_ctx, input.world_position, lighting_data.lut_ggx_emu_handle);

    for (uint i = 0; i < lighting_data.light_count; i++) {
        Moer::GLight light = light_buf.Load<Moer::GLight>(i);
        light_ctx.AccumulateLight(light, 1.0);
    }

    float3 shaded_color = light_ctx.GetResult();
    if (param.enable_extra_ambient != 0u) {
        shaded_color += param.extra_ambient_intensity * param.extra_ambient_color * brdf_ctx.albedo;
    }

    return float4(shaded_color, alpha);
}
