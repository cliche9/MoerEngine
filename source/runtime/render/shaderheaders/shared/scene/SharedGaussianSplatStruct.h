#pragma once

#ifdef CONST
#undef CONST
#endif

#ifdef __cplusplus
#define CONST constexpr
#include "misc/Traits.h"
namespace Moer {
#else
#define CONST const
namespace Moer {
#endif

struct GGaussianSplatVertex {
    float4 position;
    float4 scale_opacity;
    float4 rotation;
    float  sh[48];
};

struct GGaussianSplatBindlessHandles {
    uint vertex_buf_hdl;
    uint enabled;

    uint vertex_count;
    uint padding0;
};

#ifdef __cplusplus
static_assert(sizeof(GGaussianSplatVertex) == sizeof(float) * 60);
#endif

#ifdef __cplusplus
}
#else
}
#endif

#undef CONST
