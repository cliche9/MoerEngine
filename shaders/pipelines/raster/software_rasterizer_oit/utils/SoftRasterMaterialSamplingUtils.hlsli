/**
 * Material sampling helpers shared by software raster compute passes.
 *
 * The including shader must provide bindless texture access helpers.
 */
#pragma once

template <typename T>
T GetTextureDataLevel(int bindless_handle, float2 uv, T default_value, T missing_value) {
    if (bindless_handle >= 0) {
        return TextureHandle(bindless_handle).SampleLevel<T>(uv, 0.0);
    } else if (bindless_handle == -1) {
        return default_value;
    } else {
        return missing_value;
    }
}

float3 GetNormalFromNormalMapLevel(int normal_map, float2 uv, float3 normal, float3 tangent) {
    if (normal_map >= 0) {
        float3 normal_in_tbn = normalize((TextureHandle(normal_map).SampleLevel<float3>(uv, 0.0) * 2.0) - 1.0);
        float3 bitangent = cross(normal, tangent);
        float3x3 tbn = float3x3(tangent, bitangent, normal);
        return normalize(mul(normal_in_tbn, tbn));
    } else {
        return normal;
    }
}
