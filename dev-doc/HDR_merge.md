# HDR Merge And Image Alignment

This document describes the current HDR merge job and the image alignment step that now runs before accumulation.

Implementation files:

- `src/control/jobs/control_jobs.c`: HDR merge job, weighting, normalization, DNG write-out, integration of alignment.
- `src/common/hdr_alignment.h`: public alignment data structure and API.
- `src/common/hdr_alignment.c`: alignment estimation and CFA-aware warp application.

## Scope

The document covers two related pieces of functionality:

1. How darktable builds a merged HDR raw image from multiple bracketed raw frames.
2. How non-reference frames are geometrically aligned before being merged.

Current constraints:

- input must be raw, single-channel, `uint16` sensor data as exported through `rawprepare`
- all images must have identical dimensions and orientation
- alignment currently supports Bayer only
- output is written as a merged DNG based on the first image in the stack

## Part 1: How HDR Merge Works

### Overview

The HDR merge job processes a sequence of input images and accumulates them into two buffers:

- `pixels`: weighted radiometric sum
- `weight`: accumulated confidence / exposure weight per sensor sample

After all images have been processed, the accumulated sum is normalized by the total weight and by a stack-wide white level, then written as a DNG.

At a high level, the job does this:

1. Export each selected image through `rawprepare` into a floating-point raw mosaic.
2. Validate that the stack is compatible for merge.
3. Use the first image as the merge reference and output metadata source.
4. Optionally align each later image to the first image.
5. Compute a per-pixel merge weight from exposure and local saturation state.
6. Accumulate unclipped samples and handle saturated samples separately.
7. Normalize the final buffer.
8. Write a merged DNG and import it back into the library.

### Input Validation And Reference Frame

The first processed image initializes the merge state:

- image id of the reference frame
- Bayer / X-Trans layout used for DNG output
- image dimensions and orientation
- white-balance coefficients
- color matrix metadata
- accumulation buffers
- a copy of the first mosaic, which is also the alignment reference

The merge job rejects the stack if any later image:

- is not raw
- is not single-channel
- is not `uint16`
- differs in width or height
- differs in orientation

This is a hard constraint because the output is still a raw-domain merge, not a merge after demosaic or scene-referred resampling.

### Radiometric Calibration

For each input frame the code derives two important quantities from EXIF:

- a calibration factor `cal`
- a photon-count proxy `photoncnt`

Using aperture, exposure time, and ISO, the code computes

$$
\text{aperture area} = \pi \left(\frac{0.5 f}{N}\right)^2,
$$

$$
\mathrm{cal} = \frac{100}{\text{aperture area} \cdot t \cdot ISO},
$$

$$
\mathrm{photoncnt} = \frac{100 \cdot \text{aperture area} \cdot t}{ISO}.
$$

Interpretation:

- `cal` maps the input frame toward a common radiometric scale.
- `photoncnt` is used as a confidence prior: longer, brighter exposures contribute more where they are not clipped.

The merge state also tracks

$$
\text{whitelevel} = \max(\text{whitelevel}, \text{saturation} \cdot \mathrm{cal}),
$$

which is later used for final normalization.

### Merge Weighting

The initial merge weight for a sample is

$$
w = \mathrm{photoncnt}.
$$

That weight is then modulated by a local exposure envelope. The implementation does not decide clipping from a single pixel alone. Instead it inspects a conservative local neighborhood and computes:

- `M`: local maximum sample value
- `m`: local minimum sample value

The local maximum is fed into the envelope function `_envelope()` after a small offset:

$$
w \leftarrow w \cdot \left(\varepsilon_w + E\left(\frac{M + \text{offset}}{\text{saturation}}\right)\right),
$$

where `E(.)` is a smooth piecewise envelope and `\varepsilon_w` is a tiny floor.

The intent is:

- give high weight to well-exposed samples
- reduce weight near clipping
- still allow partially clipped neighborhoods to contribute if some channels remain useful

### Two Merge Regimes: Unclipped And Saturated

The merge logic separates the accumulation path into two cases.

#### Unclipped samples

If the local neighborhood is not considered saturated, the sample is accumulated normally:

$$
\mathrm{pixels}[k] \mathrel{+}= w \cdot in \cdot \mathrm{cal},
$$

$$
\mathrm{weight}[k] \mathrel{+}= w.
$$

This is the usual weighted merge path.

#### Saturated samples

If the local maximum indicates saturation, the sample is not added to the positive weighted sum. Instead the code keeps a fallback value only if no better unsaturated information has already been accumulated.

The logic uses negative `weight[k]` values as a marker that the pixel is currently backed by a saturation fallback instead of a proper weighted sum.

This encoding lets the merge logic represent two states in one array:

- `weight[k] > 0`: regular weighted accumulation exists
- `weight[k] <= 0`: only saturation fallback exists so far

If the entire local neighborhood is saturated, the fallback pixel may be forced to `1.0`. Otherwise the current sample is stored in a calibrated form relative to the stack white level.

This is a compact but slightly non-obvious part of the implementation.

### Interaction With Alignment

If alignment is enabled for the current image, the merge source is switched from the raw export buffer to an aligned temporary buffer.

Warped-out-of-bounds pixels are written as `-1.0f` sentinels. The merge loop checks for these sentinels and skips them:

- the center sample is ignored if it is out of bounds
- neighborhood extrema also skip out-of-bounds values

This prevents aligned edge padding from polluting the envelope or the final accumulation.

### Final Normalization

After all frames are processed, only pixels with positive accumulated weight are normalized:

$$
\mathrm{pixels}[k] \leftarrow \max\left(0, \frac{\mathrm{pixels}[k]}{\mathrm{whitelevel} \cdot \mathrm{weight}[k]}\right).
$$

This converts the accumulated sum back into a normalized raw-domain value that clips at `1.0` as expected by the DNG output path.

### Output And Metadata

The merged result is written as a DNG using metadata primarily derived from the first frame:

- filter layout
- X-Trans pattern if present
- white-balance coefficients
- camera color matrix
- EXIF blob adapted to the merged size

The output filename is derived from the first image by appending `-hdr.dng`.

After writing, darktable imports the new DNG and refreshes the collection view.

### Tradeoffs Of The Merge Design

Strengths:

- stays in the raw domain
- keeps sensor layout and camera metadata
- favors better exposed samples while preserving clipped fallback behavior
- avoids demosaic-before-merge artifacts

Weaknesses:

- merge weighting is heuristic rather than probabilistic
- saturation handling uses a compact but subtle negative-weight convention
- all frames must match exactly in geometry before merge, hence the need for alignment

## Part 2: How Image Alignment Works

### Goal

The alignment step attempts to register every non-reference image to the first image in the HDR stack before the merge weights are computed.

The current model is:

1. a global backward projective homography
2. plus a residual regularized `3x3` mesh in output space

This is a compromise between a simple rigid model and a dense local warp.

### When Alignment Runs

Alignment is integrated directly into `_control_merge_hdr_process()`:

- the first frame is copied into `ref_mosaic`
- later frames are aligned against `ref_mosaic`
- if alignment fails, the image is merged unaligned
- if the sensor is X-Trans, alignment is skipped entirely for now

The warp is only applied if the estimated transform differs meaningfully from identity.

### Design Objective And Motion Model Requirements

Before describing the alignment result type, it is useful to state what the alignment stage is trying to correct.

For HDR merge, the misalignment is usually caused by handheld capture between bracketed exposures. In practice this can include:

- image-plane translation
- small roll rotation
- weak perspective change from a slightly shifted viewpoint
- corner-local residual error that remains after a single global fit

The alignment stage is therefore not solving a completely general registration problem. It needs a motion model that is:

- expressive enough to remove visible blur across most of the frame, especially at the corners
- robust under exposure differences between bracketed frames
- cheap enough to run inside the HDR merge job
- conservative enough to avoid inventing strong local deformations on raw sensor data

In terms of geometry, the main design question is: which family of transforms is large enough to describe practical bracket misalignment, but still small enough to estimate reliably on raw-domain images?

Let the output pixel be

$$
x = \begin{bmatrix} x \\ y \end{bmatrix},
$$

and let the sampled source position be

$$
s = W(x; p) = \begin{bmatrix} s_x \\ s_y \end{bmatrix}.
$$

The transform families considered are the following.

#### (a) Translation, 2 DOF

Parameters:

$$
p = (t_x, t_y).
$$

Model:

$$
W_{trans}(x; p) =
\begin{bmatrix}
x + t_x \\
y + t_y
\end{bmatrix}.
$$

This corrects only lateral sensor-plane shift.

Why it is not enough:

- cannot model roll
- cannot model scale or shear
- leaves corner blur when the camera rotates slightly between shots

#### (b) Euclidean / rigid transform, 3 DOF

Parameters:

$$
p = (\theta, t_x, t_y).
$$

Model about the origin:

$$
W_{rigid}(x; p) =
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix}
+
\begin{bmatrix}
t_x \\
t_y
\end{bmatrix}.
$$

In the implementation, the coarse initializer applies the equivalent transform around the image center.

What it adds over translation:

- corrects image-plane shift
- corrects roll rotation

Why it is still not enough:

- assumes no scale change
- assumes no shear
- assumes no perspective effect
- improved some stacks but remained too limited for many handheld brackets

#### (c) Similarity transform, 4 DOF

Parameters:

$$
p = (a, \theta, t_x, t_y),
$$

where $a$ is a uniform scale factor.

Model:

$$
W_{sim}(x; p) =
a
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix}
+
\begin{bmatrix}
t_x \\
t_y
\end{bmatrix}.
$$

What it adds over rigid motion:

- uniform zoom in/out

Why it is still not enough:

- still cannot represent anisotropic scale or shear
- still cannot represent perspective convergence
- not a good match for the corner failures seen in practical HDR stacks

#### (d) Affine transform, 6 DOF

Parameters:

$$
p = (a_{00}, a_{01}, a_{02}, a_{10}, a_{11}, a_{12}).
$$

Model:

$$
W_{aff}(x; p) =
\begin{bmatrix}
a_{00} & a_{01} & a_{02} \\
a_{10} & a_{11} & a_{12}
\end{bmatrix}
\begin{bmatrix}
x \\
y \\
1
\end{bmatrix}.
$$

What it adds over similarity:

- anisotropic scale
- shear
- more general linear distortion

Why it is still not enough:

- affine transforms preserve parallel lines
- they cannot represent true projective effects from viewpoint change
- in practice, corner blur remained on stacks where weak perspective was present

#### (e) Homography / projective transform, 8 DOF

Parameters:

$$
p = (h_0, h_1, h_2, h_3, h_4, h_5, h_6, h_7),
$$

with implicit bottom-right element equal to $1$.

Model:

$$
W_{proj}(x; p) =
\begin{bmatrix}
\dfrac{h_0 x + h_1 y + h_2}{h_6 x + h_7 y + 1} \\
\dfrac{h_3 x + h_4 y + h_5}{h_6 x + h_7 y + 1}
\end{bmatrix}.
$$

What it adds over affine:

- perspective convergence
- the most general planar projective warp representable by one global transform

This is the current global model used by the HDR alignment code.

#### (f) Local / non-rigid warps

There is no single fixed parameter count here. The degrees of freedom depend on the parameterization: control mesh, spline coefficients, optical-flow field, or another local model.

One generic way to write a local warp is:

$$
W_{local}(x) =
\begin{bmatrix}
x \\
y
\end{bmatrix}
+
\begin{bmatrix}
u(x, y) \\
v(x, y)
\end{bmatrix},
$$

where $u(x, y)$ and $v(x, y)$ are spatially varying displacement fields.

What it adds over homography:

- can model parallax, lens mismatch, and scene-local deformation

Why it was not chosen as the primary model:

- more expensive to estimate and apply
- more likely to overfit low-texture raw data
- harder to keep stable under large exposure differences
- riskier for raw-domain processing because local distortions can create implausible geometry

### Why The Design Uses A Layered Approach

The alignment pipeline uses different motion models at different stages:

1. **ECC refinement** uses option (b), a native 3-DOF Euclidean model (rotation + translation). The optimizer builds its Hessian and Jacobian entirely in the $(\theta, t_x, t_y)$ parameter space, so no energy can leak into shear, scale, or perspective DOFs.

2. **Adaptive DOF escalation** (at the finest pyramid level only): if the 3-DOF fit is insufficient, the pipeline selectively escalates to option (d) 6-DOF affine and then option (e) 8-DOF projective ECC, starting from the stable 3-DOF result. Each escalation step is gated by improvement, conditioning, and sanity checks.

3. **Corner refinement** uses option (e), computing a local 4-point homography correction from NCC patches at the four image corners. This absorbs weak perspective and residual scale that the preceding ECC model did not capture.

4. **Mesh residuals** add a small amount of option (f) behavior via a regularized $3\times3$ local displacement grid.

The result is stored as an 8-DOF backward homography (the common representation used by corner refinement and the final warp). The core iterative optimizer starts with only 3 parameters per step and selectively escalates to 6 or 8 when the data supports it.

#### Why ECC starts with the Euclidean model

An earlier version of the code ran ECC with the full 8-DOF projective Jacobian. This worked adequately on easy stacks but failed systematically on difficult HDR brackets (e.g. very dark + very bright exposures). The failure mode was:

- gradient noise in the 8-DOF normal equations coupled rotation with shear and perspective
- the Gauss-Newton update drifted the off-diagonal and diagonal elements away from a valid rotation
- post-hoc clamping or projection back onto the Euclidean subspace did not help, because the 8×8 Hessian inverse had already mixed the DOFs

Projecting the 8-DOF solution back onto 3-DOF after solving is not equivalent to natively solving 3-DOF. The 8×8 Hessian inverse couples all 8 parameters, so the projected update direction is contaminated by the 5 excess DOFs.

The native 3-DOF solver eliminates this problem entirely: the 3×3 Hessian only has Euclidean directions, so every update is guaranteed to stay on the rigid-body manifold.

The adaptive DOF escalation then introduces higher-DOF models only at the finest level, starting from the stable 3-DOF solution, and only accepts the result when improvement, conditioning, and sanity checks all pass. This avoids the instability problems of running higher-DOF ECC throughout the pyramid while still capturing perspective and scale corrections when the data supports them.

#### Why higher-DOF models are not used during the pyramid

Options (c) through (e) were all tried as ECC models during the multi-resolution pyramid and found to be increasingly unstable under large exposure differences:

- Similarity (4 DOF): scale DOF absorbed gradient magnitude differences as geometric zoom
- Affine (6 DOF): shear and anisotropic scale DOFs drifted under noise
- Projective (8 DOF): perspective terms amplified instability, especially at coarse pyramid levels

The fundamental issue is that ECC on gradient-magnitude images of differently-exposed brackets produces a Hessian with unfavorable condition numbers in the non-rigid directions. Restricting the pyramid solver to 3-DOF avoids this entirely.

The adaptive escalation at the finest level sidesteps these problems because: (a) it starts from an already-converged 3-DOF solution, (b) it runs only at fine resolution where gradients are well-resolved, and (c) it is gated by conditioning and sanity checks that reject unstable solutions.

Option (f), a general local warp, would be even more expressive but is too large a jump in complexity and risk for the current raw-domain HDR merge path.

### Alignment Data Model

The public result type is `dt_hdr_alignment_t`:

```c
typedef struct dt_hdr_alignment_t
{
  float H[8];
  float mesh_dx[DT_HDR_ALIGN_MESH_NODES];
  float mesh_dy[DT_HDR_ALIGN_MESH_NODES];
} dt_hdr_alignment_t;
```

Interpretation:

- `H[8]` is an 8-DOF backward homography with implicit `h_{22} = 1`
- `mesh_dx[]` and `mesh_dy[]` are full-resolution output-space residuals on a regular `3x3` grid
- nodes are stored in row-major order across the image

### Alignment Pipeline

For one candidate frame the estimator does the following:

1. Reduce the Bayer mosaic to half-resolution grayscale by averaging each `2x2` Bayer block.
2. Build a `2x` pyramid down to a coarse image of about `64` pixels on the longest side.
3. Normalize each pyramid level to `[0, 1]` so gradient magnitudes from dark and bright exposures have comparable numerical scale.
4. Run an exhaustive Euclidean search for translation and roll using NCC at the coarsest level.
5. Convert that coarse Euclidean result into a backward homography.
6. Refine the Euclidean parameters from coarse to fine using native 3-DOF weighted ECC on gradient-magnitude images.
7. At the finest level, adaptively escalate to 6-DOF affine or 8-DOF projective ECC if the 3-DOF fit is insufficient (see _Adaptive DOF Escalation_ below).
8. At the finest levels, run an additional corner-focused NCC correction pass that fits a local 4-point homography.
9. Estimate local residual shifts on a small `3x3` grid of patches.
10. Regularize that residual field with neighbor smoothness.
11. Convert the final grayscale-level homography back to full-resolution Bayer coordinates.

### Why The Estimator Uses Grayscale Bayer Blocks

The estimator does not run directly on the mosaiced image because raw Bayer phases contain structured color sampling differences that are not geometric motion.

Reducing each `2x2` Bayer cell to one grayscale sample:

- reduces computation by `4x`
- suppresses Bayer-phase artifacts in the estimator
- avoids requiring a demosaic stage inside HDR merge

The downside is that some fine chromatic structure is lost, which is why the final warp is still applied in a CFA-aware way later.

### Coarse Initialization

At the coarsest pyramid level the code searches over:

- roll angle in `[-10°, +10°]` with `0.5°` steps
- translation in a radius proportional to image size

This stage uses plain NCC, not ECC, because:

- the images are very small
- exhaustive search is cheap there
- ECC needs a reasonable initialization to converge reliably

This coarse stage is intentionally limited to translation plus roll. The Euclidean ECC refinement then takes over from this initial estimate.

### ECC Refinement

The main optimizer is a forward-additive ECC-style update using a native 3-DOF Euclidean model. Each ECC iteration solves directly for $(\Delta\theta, \Delta t_x, \Delta t_y)$ by building a 3×3 Hessian in the Euclidean parameter space.

Important implementation choices:

- the feature images are gradient magnitudes, not raw intensities
- the current rotation angle $\theta$ is extracted from the normalized homography via $\cos\theta = H_n[0]$, $\sin\theta = H_n[1]$
- the Jacobian is a 3-column matrix mapping $(\Delta\theta, \Delta t_x, \Delta t_y)$ to pixel intensity changes
- the normal equations are solved as a dense 3×3 system with inline Gauss-Jordan elimination
- Tikhonov regularization ($\lambda = 0.01 \cdot \text{trace}/3$) is added to the Hessian diagonal
- updates are clipped before being applied
- the angle is updated exactly via $\theta_{new} = \text{atan2}(\sin\theta, \cos\theta) + \Delta\theta$, avoiding linearization error
- the result is written back as a Euclidean homography ($h_6 = h_7 = 0$, diagonal = $\cos\theta$, off-diagonal = $\pm\sin\theta$)
- outer image regions receive extra weight to keep corners relevant

Why the native 3-DOF formulation:

- an earlier 8-DOF projective ECC solver failed on difficult HDR brackets because gradient noise in the 5 excess DOFs corrupted the rotation and translation estimates
- projecting an 8-DOF solution onto the 3-DOF Euclidean subspace after solving does not help, because the 8×8 Hessian inverse already mixes all DOFs
- building the Hessian natively in $(\theta, t_x, t_y)$ space guarantees that every update stays on the rigid-body manifold

Why gradient magnitude is used:

- HDR brackets differ in exposure, so raw intensity correlation is unstable
- gradient magnitude removes much of the exposure-dependent DC and scale variation
- edge structure survives better across the bracketed stack

Why normalized coordinates are used:

- translation parameters remain on a unit scale regardless of image resolution
- the normalized form makes per-step clamps resolution-independent

### Corner Refinement And Residual Mesh

Even a full homography can leave visible corner blur on some stacks. The implementation therefore applies two corner-oriented stages:

1. a small fine-level homography correction estimated from four corner patches
2. a final residual regularized `3x3` mesh used during warp application

The mesh is not another global homography. It is a low-order local correction that shifts output coordinates before the homography is evaluated.

This design targets the observed failure mode directly: a globally decent fit with local corner residuals.

### Mathematical Formulation

This section describes the exact parameterization used by `hdr_alignment.c`.

#### Backward homography

The global warp is stored as

$$
H =
\begin{bmatrix}
h_0 & h_1 & h_2 \\
h_3 & h_4 & h_5 \\
h_6 & h_7 & 1
\end{bmatrix}.
$$

For an output coordinate $(x, y)$, the sampled source coordinate is

$$
d = h_6 x + h_7 y + 1,
$$

$$
s_x = \frac{h_0 x + h_1 y + h_2}{d},
\qquad
s_y = \frac{h_3 x + h_4 y + h_5}{d}.
$$

This is the backward-mapping form used both by the estimator and by the final warp.

#### Bayer reduction

The grayscale estimator image is

$$
G(x, y) = \frac{1}{4}\sum_{i=0}^{1}\sum_{j=0}^{1} M(2x+i, 2y+j).
$$

At the end of estimation, grayscale coordinates are mapped back to full-resolution Bayer coordinates with the block-center convention

$$
x_f = 2x_g + 0.5,
\qquad
y_f = 2y_g + 0.5.
$$

#### Euclidean initializer

The coarse search estimates $(t_x, t_y, \theta)$ first. With image center $(c_x, c_y)$, those parameters are converted into a backward homography

$$
H_E =
\begin{bmatrix}
\cos\theta & \sin\theta & -\cos\theta\,(t_x + c_x) - \sin\theta\,(t_y + c_y) + c_x \\
-\sin\theta & \cos\theta & \sin\theta\,(t_x + c_x) - \cos\theta\,(t_y + c_y) + c_y \\
0 & 0 & 1
\end{bmatrix}.
$$

#### Normalized coordinates

Before each ECC update, the pixel-space homography is converted into centered normalized coordinates. Let

$$
s = \frac{1}{2}\max(w-1, h-1),
\qquad
c_x = \frac{w-1}{2},
\qquad
c_y = \frac{h-1}{2},
$$

and

$$
A =
\begin{bmatrix}
s & 0 & c_x \\
0 & s & c_y \\
0 & 0 & 1
\end{bmatrix}.
$$

Then

$$
H_n = A^{-1} H_{pixel} A.
$$

This keeps the projective parameters on a usable scale.

#### Feature image

ECC is run on gradient magnitude, not raw intensity. With Sobel derivatives $(g_x, g_y)$,

$$
F(x, y) = \sqrt{g_x(x, y)^2 + g_y(x, y)^2}.
$$

The reference feature image is $T(x)$ and the warped candidate feature image is $I_H(x)$.

#### Weighted ECC objective

The implementation gives extra influence to outer image regions. In normalized coordinates $(x_n, y_n)$, the spatial weight is

$$
w(x_n, y_n) = 1 + \lambda \min\left(1, \frac{x_n^2 + y_n^2}{2}\right),
$$

where $\lambda = \texttt{HDR\_ALIGN\_ECC\_EDGE\_WEIGHT}$.

Using weighted zero-mean signals

$$
t(x) = T(x) - \mu_T,
\qquad
i(x) = I_H(x) - \mu_I,
$$

the weighted ECC score is

$$
\rho = \frac{\sum_{x \in \Omega} w(x) \, t(x) \, i(x)}
{\sqrt{\sum_{x \in \Omega} w(x) \, t(x)^2} \, \sqrt{\sum_{x \in \Omega} w(x) \, i(x)^2}}.
$$

#### 3-DOF Euclidean Jacobian

The ECC solver operates in the 3-DOF Euclidean parameter space $(\theta, t_x, t_y)$ where $\theta$ is the rotation angle about the image center and $(t_x, t_y)$ are translations in normalized coordinates.

Let $(g_x, g_y)$ be the spatial gradient of the current warped feature image at pixel $(x, y)$, and let $(x_n, y_n)$ be the normalized coordinates:

$$
x_n = \frac{x - c_x}{s}, \qquad y_n = \frac{y - c_y}{s}.
$$

The 3-column row Jacobian used in the implementation is

$$
J =
\begin{bmatrix}
J_\theta, & J_{t_x}, & J_{t_y}
\end{bmatrix},
$$

where

$$
J_\theta = s \bigl( g_x (-\sin\theta \, x_n + \cos\theta \, y_n) + g_y (-\cos\theta \, x_n - \sin\theta \, y_n) \bigr),
$$

$$
J_{t_x} = s \, g_x,
\qquad
J_{t_y} = s \, g_y.
$$

The $J_\theta$ term is the derivative of the warped image intensity with respect to a rotation about the image center. It couples both gradient components through the current rotation angle, correctly capturing the non-linear rotation manifold.

#### Linearized ECC update

The implementation projects away the photometric scale direction using the standard ECC formulation and solves a weighted normal equation of the form

$$
\left(\sum_{x \in \Omega} w(x) \, J'(x)^T J'(x)\right) \Delta p
= \sum_{x \in \Omega} w(x) \, J'(x)^T e(x),
$$

with

$$
\Delta p = [\Delta\theta, \Delta t_x, \Delta t_y]^T.
$$

Here $J'$ is the Jacobian after subtracting the mean and projecting out the component along the current warped image (the standard ECC photometric normalization). The system is a 3×3 dense linear system, solved with inline Gauss-Jordan elimination and partial pivoting.

Tikhonov regularization is applied to the Hessian diagonal:

$$
H_{kk} \leftarrow H_{kk} + \lambda, \qquad \lambda = \frac{0.01}{3} \operatorname{tr}(H).
$$

This prevents singular or near-singular systems on low-texture images without biasing the solution significantly.

#### Angle update

The rotation angle is updated exactly rather than via the usual additive linearization:

$$
\theta_{new} = \operatorname{atan2}(\sin\theta, \cos\theta) + \Delta\theta,
$$

$$
H_n[0] = H_n[4] = \cos(\theta_{new}), \qquad H_n[1] = \sin(\theta_{new}), \qquad H_n[3] = -\sin(\theta_{new}).
$$

This avoids accumulating linearization error in the rotation over many iterations, which matters because the coarse initialization can be several degrees away from the optimum.

The perspective elements are forced to zero: $H_n[6] = H_n[7] = 0$. This ensures the homography remains exactly Euclidean throughout the ECC pyramid. Perspective and other non-rigid corrections are handled by the subsequent corner refinement and mesh stages.

#### Trust region and clamps

To keep the optimizer from taking catastrophic steps, the update is clipped per component before being applied:

$$
|\Delta\theta| \leq 0.01 \;(\approx 0.57°), \qquad |\Delta t_x|, |\Delta t_y| \leq 0.10.
$$

The translation clamp is in normalized coordinates, so $0.10$ corresponds to $10\%$ of the image half-diagonal per iteration. No post-hoc parameter clamping is needed because the 3-DOF parameterization cannot produce shear, scale drift, or perspective artifacts.

#### Residual regularized mesh

After homography refinement, the code estimates residual translations on a regular `3x3` patch grid over the image. Let the node values be

$$
(\Delta x_{r,c}, \Delta y_{r,c}),
\qquad
r \in \{0,1,2\}, \; c \in \{0,1,2\}.
$$

These node estimates are not used raw. They are regularized by repeatedly averaging each node with its 4-neighborhood while keeping valid patch measurements as data anchors. Conceptually, one smoothing step has the form

$$
\Delta x_{r,c}^{new} =
\frac{w_{data} \Delta x_{r,c}^{raw} + \lambda \sum_{n \in \mathcal{N}(r,c)} \Delta x_n}
{w_{data} + \lambda |\mathcal{N}(r,c)|},
$$

and likewise for $\Delta y$, where $\mathcal{N}(r,c)$ denotes the 4-neighbors of the node.

For a full-resolution output point $(x_f, y_f)$, normalized image coordinates are

$$
u = \frac{x_f}{W-1},
\qquad
v = \frac{y_f}{H-1}.
$$

The regular mesh is then sampled piecewise bilinearly. If

$$
g_x = u (N_x - 1),
\qquad
g_y = v (N_y - 1),
$$

with $N_x = N_y = 3$, then the active cell is determined by

$$
i = \lfloor g_x \rfloor,
\qquad
j = \lfloor g_y \rfloor,
$$

and local cell coordinates

$$
\alpha = g_x - i,
\qquad
\beta = g_y - j.
$$

The residual displacement inside that cell is

$$
\Delta x_{mesh} = (1-\alpha)(1-\beta)\Delta x_{j,i}
+ \alpha(1-\beta)\Delta x_{j,i+1}
+ (1-\alpha)\beta\Delta x_{j+1,i}
+ \alpha\beta\Delta x_{j+1,i+1},
$$

$$
\Delta y_{mesh} = (1-\alpha)(1-\beta)\Delta y_{j,i}
+ \alpha(1-\beta)\Delta y_{j,i+1}
+ (1-\alpha)\beta\Delta y_{j+1,i}
+ \alpha\beta\Delta y_{j+1,i+1}.
$$

The homography is then evaluated at

$$
x_w = x_f + \Delta x_{mesh},
\qquad
y_w = y_f + \Delta y_{mesh}.
$$

This is why the mesh acts as a residual output-space correction layered on top of the global projective warp.

### CFA-Aware Warp Application

The final warp is not applied directly to the mosaiced image. Instead the Bayer mosaic is decomposed into four half-resolution phase planes.

For one plane-local output coordinate $(x_p, y_p)$ with phase offset $(o_x, o_y) \in \{0,1\}^2$,

$$
x_f = 2x_p + o_x,
\qquad
y_f = 2y_p + o_y.
$$

After the mesh-adjusted homography produces a full-resolution source coordinate $(s_x^{full}, s_y^{full})$, that source point is mapped back to plane-local coordinates:

$$
s_x^{plane} = \frac{s_x^{full} - o_x}{2},
\qquad
s_y^{plane} = \frac{s_y^{full} - o_y}{2}.
$$

That plane-local source coordinate is bilinearly sampled in the corresponding Bayer phase plane.

This prevents the warp from mixing different CFA phases.

### Tradeoffs Of The Alignment Design

Strengths:

- works directly on raw-domain data
- much more robust to exposure changes than pure raw-intensity NCC
- native 3-DOF ECC is highly stable even under extreme exposure differences
- adaptive DOF escalation adds scale / shear / perspective correction only when the data supports it
- layered architecture: rigid ECC → adaptive escalation → corner homography → local mesh, each stage adds flexibility incrementally
- preserves CFA phase during the final warp
- residual regularized mesh improves difficult corner cases at low additional cost

Weaknesses:

- Bayer only
- local residual correction is only low-order bilinear
- strong parallax or non-planar scenes can still leave blur
- relies on heuristic search ranges, weights, and trust-region clamps
- escalation threshold and gating constants are empirical

### Why This Design Was Chosen

Multiple alternatives were tested iteratively:

- 8-DOF projective ECC: unstable on difficult HDR brackets — gradient noise in the excess DOFs caused shear drift and angle erosion
- 8-DOF ECC with post-hoc rotation enforcement: still failed because the 8×8 Hessian inverse mixed all DOFs before clamping
- 8-DOF ECC with projection onto 3-DOF subspace: failed because projecting the update vector after an 8×8 solve is not equivalent to natively solving 3-DOF
- native 3-DOF Euclidean ECC: stable and correct — the optimizer can only move in $(\theta, t_x, t_y)$ directions

The current layered design (3-DOF ECC + adaptive DOF escalation + 4-point corner homography + mesh residuals) keeps the ECC solver maximally robust while still handling the full range of practical misalignment through the subsequent correction stages.

### Adaptive DOF Escalation

After the 3-DOF Euclidean ECC pyramid converges at the finest level, the alignment pipeline measures the weighted ECC score $\rho$ and decides whether the rigid fit is sufficient or whether higher-DOF models should be attempted.

#### Motivation

The 3-DOF Euclidean model is maximally stable under large exposure differences, but it can only correct rotation and translation. When the true misalignment includes weak perspective or anisotropic scale (e.g. from a slightly shifted viewpoint between brackets), the rigid model leaves residual blur that the subsequent corner refinement and mesh stages may not fully absorb.

Adaptive DOF escalation addresses this by selectively introducing additional degrees of freedom only when the data supports them.

#### Pipeline

The escalation runs once, at pyramid level 0 (the finest grayscale level), between 3-DOF ECC convergence and corner refinement:

1. **Measure baseline quality**: compute the weighted ECC score $\rho_{3}$ of the converged 3-DOF result on gradient-magnitude images. If $\rho_{3} \geq \rho_{thresh}$ (currently $0.85$), the rigid fit is considered sufficient and escalation is skipped entirely.

2. **Try 6-DOF affine**: starting from the 3-DOF homography, run a limited number of forward-additive ECC iterations using a native 6-DOF affine Jacobian (scale, shear, and translation, but no perspective). Accept if:
   - the refined $\rho_{6} > \rho_{3}$ with improvement $\geq \Delta\rho_{min}$
   - the Hessian condition number is below the maximum allowed
   - the result passes geometric sanity checks

3. **Try 8-DOF projective**: starting from the better of the 3-DOF or 6-DOF result, run a limited number of ECC iterations with the full 8-DOF projective Jacobian. Accept under the same gating criteria as step 2.

4. **Select best**: the pipeline accepts the highest-DOF result that passed all checks. If neither escalation step improved $\rho$, the original 3-DOF result is kept unchanged.

#### Mathematical Formulation

##### 6-DOF Affine Jacobian

For the affine model in normalized coordinates, the warp is

$$
W_n(x_n, y_n) =
\begin{bmatrix}
H_n[0] x_n + H_n[1] y_n + H_n[2] \\
H_n[3] x_n + H_n[4] y_n + H_n[5]
\end{bmatrix},
$$

with $H_n[6] = H_n[7] = 0$. The 6-column Jacobian mapping parameter updates to intensity changes is

$$
J = \begin{bmatrix}
s \, g_x x_n, & s \, g_x y_n, & s \, g_x, &
s \, g_y x_n, & s \, g_y y_n, & s \, g_y
\end{bmatrix},
$$

where $(g_x, g_y)$ are the spatial gradients of the warped feature image and $s$ is the normalization scale. The parameters are updated additively:

$$
H_n[k] \leftarrow H_n[k] + \Delta p_k, \qquad k \in \{0, \ldots, 5\}.
$$

##### 8-DOF Projective Jacobian

For the full projective model, let

$$
d = H_n[6] x_n + H_n[7] y_n + 1,
\qquad
s_{x,n} = H_n[0] x_n + H_n[1] y_n + H_n[2],
\qquad
s_{y,n} = H_n[3] x_n + H_n[4] y_n + H_n[5].
$$

The 8-column Jacobian is

$$
J_k = s \, g_x \frac{\partial W_x}{\partial H_n[k]} + s \, g_y \frac{\partial W_y}{\partial H_n[k]},
$$

with the first 6 columns identical to the affine case scaled by $1/d$, and the perspective columns

$$
J_6 = -s \frac{(g_x s_{x,n} + g_y s_{y,n}) \, x_n}{d^2},
\qquad
J_7 = -s \frac{(g_x s_{x,n} + g_y s_{y,n}) \, y_n}{d^2}.
$$

#### Gating Criteria

Each escalation step is gated by three checks:

1. **Improvement gate**: the new $\rho$ must exceed the baseline by at least $\Delta\rho_{min}$ (`HDR_ALIGN_ESCALATION_MIN_IMPROVEMENT`, currently $0.01$).

2. **Conditioning gate**: during the Gauss-Jordan solve, the ratio of the largest to smallest absolute pivot values is tracked. If this exceeds `HDR_ALIGN_ESCALATION_MAX_COND` ($10^6$), the extra parameters are considered ill-determined and the result is rejected.

3. **Sanity gate**: the escalated homography must pass `_homography_is_sane_escalated`, which checks:
   - diagonal elements of the normalized homography are within $[0.85, 1.15]$ (allows ~15% scale)
   - off-diagonal elements are within $[-0.25, 0.25]$ (allows moderate shear)
   - for affine: perspective terms remain exactly zero
   - for projective: perspective terms remain within $[-0.02, 0.02]$
   - all four image corners map to source positions within 15% of the image diagonal

#### Trust Region and Clamps

The higher-DOF ECC iterations use per-component clamps to prevent catastrophic steps:

| Parameter group   | Max update per iteration |
|---|---|
| Affine diagonal / off-diagonal ($H_n[0,1,3,4]$) | $0.02$ |
| Translation ($H_n[2,5]$) | $0.10$ (normalized) |
| Perspective ($H_n[6,7]$, 8-DOF only) | $0.001$ |

Tikhonov regularization is applied to the Hessian diagonal with the same $\lambda = 0.01 \cdot \operatorname{tr}(H) / n$ formula as the 3-DOF solver, scaled to the appropriate dimension.

#### Interaction with Corner Refinement

Corner refinement still runs after DOF escalation. In practice:

- If escalation absorbed the perspective / scale error, the corner NCC patches will report near-zero shifts and the corner refinement stage will be a no-op.
- If escalation improved the global fit but left local corner residuals, corner refinement will absorb those residuals as before.
- If escalation was skipped (high $\rho_{3}$), the pipeline is identical to the previous 3-DOF-only design.

No changes to the corner refinement acceptance thresholds were needed.

#### Configuration Constants

| Constant | Value | Purpose |
|---|---|---|
| `HDR_ALIGN_ESCALATION_RHO_THRESHOLD` | $0.85$ | Minimum 3-DOF $\rho$ below which escalation is attempted |
| `HDR_ALIGN_ESCALATION_MIN_IMPROVEMENT` | $0.01$ | Minimum $\Delta\rho$ to accept a higher-DOF result |
| `HDR_ALIGN_ESCALATION_MAX_COND` | $10^6$ | Maximum Hessian condition number for acceptance |
| `HDR_ALIGN_ESCALATION_MAX_ITER` | $30$ | Maximum ECC iterations per escalation stage |
| `HDR_ALIGN_ESCALATION_EPSILON` | $5 \times 10^{-3}$ | Convergence threshold for escalated ECC |

These values were chosen conservatively. The threshold $\rho = 0.85$ means escalation only activates on stacks where the rigid fit leaves significant residual misalignment. The improvement gate ($0.01$) prevents accepting results that are numerically but not visually better.

## Future Work

The adaptive DOF escalation feature could be further refined with:

- **Empirical threshold tuning**: the current $\rho_{thresh} = 0.85$ is conservative. Testing on a diverse corpus of HDR stacks could inform a better default or an adaptive threshold based on stack properties (number of frames, exposure range).
- **Per-level escalation**: currently escalation runs only at the finest pyramid level. Running it at intermediate levels (gated by the same improvement and conditioning checks) could help stacks where the 3-DOF fine-level result is locally optimal in the wrong basin.

## File Map

- `src/control/jobs/control_jobs.c`: HDR merge job and alignment integration
- `src/common/hdr_alignment.h`: public alignment structure and API
- `src/common/hdr_alignment.c`: estimator, warp application, Bayer plane handling
