/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Blend-mode constants shared by the Vulkan blendop kernels
    (blendop_lab.cl / blendop_rgb_hsl.cl / blendop_rgb_jzczhz.cl).

    Values mirror dt_develop_blend_mode_t in src/develop/blend.h and
    the enum copy at the top of data/kernels/blendop.cl — keep all
    three in sync. The GLSL twins (.comp) inline the same values as
    #defines since GLSL has no #include.
*/

#ifndef DT_VULKAN_BLENDOP_H
#define DT_VULKAN_BLENDOP_H

#define DEVELOP_BLEND_LIGHTEN          0x02
#define DEVELOP_BLEND_DARKEN           0x03
#define DEVELOP_BLEND_MULTIPLY         0x04
#define DEVELOP_BLEND_AVERAGE          0x05
#define DEVELOP_BLEND_ADD              0x06
#define DEVELOP_BLEND_SUBTRACT         0x07
#define DEVELOP_BLEND_DIFFERENCE       0x08
#define DEVELOP_BLEND_SCREEN           0x09
#define DEVELOP_BLEND_OVERLAY          0x0A
#define DEVELOP_BLEND_SOFTLIGHT        0x0B
#define DEVELOP_BLEND_HARDLIGHT        0x0C
#define DEVELOP_BLEND_VIVIDLIGHT       0x0D
#define DEVELOP_BLEND_LINEARLIGHT      0x0E
#define DEVELOP_BLEND_PINLIGHT         0x0F
#define DEVELOP_BLEND_LIGHTNESS        0x10
#define DEVELOP_BLEND_CHROMA           0x11 /* blend.h: DEVELOP_BLEND_CHROMATICITY */
#define DEVELOP_BLEND_HUE              0x12
#define DEVELOP_BLEND_COLOR            0x13
#define DEVELOP_BLEND_COLORADJUST      0x16
#define DEVELOP_BLEND_DIFFERENCE2      0x17
#define DEVELOP_BLEND_NORMAL2          0x18
#define DEVELOP_BLEND_BOUNDED          0x19
#define DEVELOP_BLEND_LAB_LIGHTNESS    0x1A
#define DEVELOP_BLEND_LAB_COLOR        0x1B
#define DEVELOP_BLEND_HSV_LIGHTNESS    0x1C /* blend.h: DEVELOP_BLEND_HSV_VALUE */
#define DEVELOP_BLEND_HSV_COLOR        0x1D
#define DEVELOP_BLEND_LAB_L            0x1E
#define DEVELOP_BLEND_LAB_A            0x1F
#define DEVELOP_BLEND_LAB_B            0x20
#define DEVELOP_BLEND_RGB_R            0x21
#define DEVELOP_BLEND_RGB_G            0x22
#define DEVELOP_BLEND_RGB_B            0x23
#define DEVELOP_BLEND_SUBTRACT_INVERSE 0x25
#define DEVELOP_BLEND_DIVIDE           0x26
#define DEVELOP_BLEND_DIVIDE_INVERSE   0x27
#define DEVELOP_BLEND_GEOMETRIC_MEAN   0x28
#define DEVELOP_BLEND_HARMONIC_MEAN    0x29

#define DEVELOP_BLEND_REVERSE          0x80000000u
#define DEVELOP_BLEND_MODE_MASK        0xFFu

#endif // DT_VULKAN_BLENDOP_H
