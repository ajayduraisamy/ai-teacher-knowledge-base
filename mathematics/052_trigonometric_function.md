# Concept: Trigonometric Function

## Concept ID

MATH-052

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Functions

## Learning Objectives

- Define the six trigonometric functions using the unit circle and right triangles.
- Convert between degrees and radians and understand radian measure.
- Analyze periodicity, amplitude, and phase shift of trigonometric functions.
- Apply fundamental identities: $\sin^2 \theta + \cos^2 \theta = 1$, sum/difference formulas.
- Compute derivatives of sine and cosine: $\frac{d}{dx} \sin x = \cos x$, $\frac{d}{dx} \cos x = -\sin x$.
- Connect trigonometric functions to AI/ML concepts: positional encoding in transformers, Fourier features for NeRF, and periodic activations in SIREN networks.

## Prerequisites

- Understanding of functions (MATH-044), domain (MATH-045), and range (MATH-046).
- Familiarity with the Cartesian coordinate plane and the Pythagorean theorem.
- Basic geometry: angles, circles, and triangles.
- Exponential functions (MATH-050) for understanding Euler's formula.

## Definition

Trigonometric functions relate angles to ratios of sides in right triangles and to coordinates on the unit circle. The six primary trigonometric functions are:

**Sine:** $\sin \theta = \frac{\text{opposite}}{\text{hypotenuse}} = \frac{y}{r}$
**Cosine:** $\cos \theta = \frac{\text{adjacent}}{\text{hypotenuse}} = \frac{x}{r}$
**Tangent:** $\tan \theta = \frac{\text{opposite}}{\text{adjacent}} = \frac{y}{x} = \frac{\sin \theta}{\cos \theta}$
**Cosecant:** $\csc \theta = \frac{1}{\sin \theta}$
**Secant:** $\sec \theta = \frac{1}{\cos \theta}$
**Cotangent:** $\cot \theta = \frac{1}{\tan \theta} = \frac{\cos \theta}{\sin \theta}$

For the unit circle (radius $r = 1$), these simplify to $\sin \theta = y$ and $\cos \theta = x$, where $(x, y)$ is the point on the circle corresponding to angle $\theta$.

The domain of $\sin \theta$ and $\cos \theta$ is $\mathbb{R}$ (all real numbers), and their range is $[-1, 1]$. The domain of $\tan \theta$ excludes $\theta = \frac{\pi}{2} + n\pi$ (where $\cos \theta = 0$).

## Intuition

Imagine a point moving counterclockwise around a circle of radius 1 centered at the origin. As the point travels, its vertical coordinate traces $\sin \theta$ and its horizontal coordinate traces $\cos \theta$. The angle $\theta$ measures how far the point has traveled along the circle.

Starting at $(1, 0)$ (angle 0):
- At $\theta = \frac{\pi}{2}$ (90 degrees), the point is at $(0, 1)$: $\cos = 0$, $\sin = 1$.
- At $\theta = \pi$ (180 degrees), the point is at $(-1, 0)$: $\cos = -1$, $\sin = 0$.
- At $\theta = \frac{3\pi}{2}$ (270 degrees), the point is at $(0, -1)$: $\cos = 0$, $\sin = -1$.
- At $\theta = 2\pi$ (360 degrees), the point returns to $(1, 0)$: one full cycle.

The motion is **periodic**: after $2\pi$ radians (one full revolution), the coordinates repeat. This periodicity is the defining characteristic of trigonometric functions.

For the right triangle interpretation, consider a right triangle with angle $\theta$. The sine is the ratio of the side opposite $\theta$ to the hypotenuse. The cosine is the ratio of the adjacent side to the hypotenuse. These ratios depend only on the angle, not the size of the triangle.

## Why This Concept Matters

Trigonometric functions are fundamental to describing oscillatory and periodic phenomena, which appear throughout science and engineering:

1. **Physics:** Waves (sound, light, water), simple harmonic motion (springs, pendulums), alternating current (AC) electricity, and quantum mechanics (wave functions).

2. **Engineering:** Signal processing (Fourier transforms), navigation (GPS, celestial navigation), robotics (inverse kinematics), and structural engineering (vibration analysis).

3. **Computer Graphics:** 3D rotations using rotation matrices, animation curves, and procedural generation of natural phenomena.

4. **Machine Learning:** Positional encoding in transformers, Fourier features for high-frequency detail learning, periodic activation functions, and attention mechanisms.

## Historical Background

Trigonometry has ancient origins. The Egyptians and Babylonians used basic trigonometric ratios for surveying and astronomy as early as 2000 BCE. The Greek astronomer Hipparchus (c. 150 BCE) created the first known trigonometric table, a chord table for constructing circles.

The name "trigonometry" comes from Greek *trigonon* (triangle) and *metron* (measure). Indian mathematicians (Aryabhata, Brahmagupta) developed sine and cosine functions in the 5th-7th centuries CE. The terms "sine" and "cosine" have a complex etymological history: "sine" comes from the Sanskrit *jya-ardha* (half-chord), which was translated to Arabic *jiba*, then mistakenly to Latin *sinus* (meaning "bay" or "curve").

Islamic scholars like Al-Battani and Abu al-Wafa refined trigonometric tables and discovered key identities. European mathematicians (Regionontanus, Copernicus, Rheticus) developed trigonometry as a distinct discipline in the 15th-16th centuries.

The modern analytic approach — defining $\sin$ and $\cos$ as functions of real numbers using the unit circle — was developed by Euler in the 18th century. Euler also discovered the remarkable formula $e^{i\theta} = \cos\theta + i\sin\theta$, connecting trigonometry to complex analysis.

Fourier's work (1822) on representing periodic functions as sums of sines and cosines revolutionized mathematics and physics. The Fourier series and Fourier transform remain essential tools in signal processing, data analysis, and modern machine learning.

## Real World Examples

**Example 1: Simple Harmonic Motion.** A mass on a spring oscillates according to $x(t) = A \cos(\omega t + \phi)$, where $A$ is amplitude, $\omega$ is angular frequency, and $\phi$ is phase. For $A = 0.5$ m, $\omega = 2\pi$ rad/s (period 1 s), $\phi = 0$, the position at $t = 0.25$ s is $x = 0.5 \cos(\pi/2) = 0$ m (passing through equilibrium).

**Example 2: AC Electricity.** Voltage in a wall outlet follows $V(t) = V_0 \sin(2\pi ft)$ where $f = 60$ Hz (in the US) or $f = 50$ Hz (in Europe). For $V_0 = 170$ V (peak voltage), $V(t) = 170 \sin(120\pi t)$.

**Example 3: Sound Waves.** A pure tone (sine wave) at frequency $f = 440$ Hz (A4 note) is $p(t) = A \sin(2\pi \cdot 440 \cdot t)$. Musical instruments produce complex sounds that are sums of many sine waves at different frequencies (harmonics).

**Example 4: GPS Navigation.** GPS uses trilateration based on distances to satellites. The satellites broadcast their positions and times. The receiver solves for its position using trigonometric relationships between angles and distances.

**Example 5: Daylight Hours.** The number of daylight hours as a function of day of year approximately follows a sine wave: $D(t) = 12 + A \sin(\frac{2\pi}{365}(t - \phi))$, where $A$ depends on latitude (larger $A$ means more variation).

## AI/ML Relevance

Trigonometric functions have become increasingly important in modern deep learning through several key applications:

**1. Positional Encoding in Transformers.** The transformer architecture (Vaswani et al., 2017) uses sinusoidal positional encodings to give the model information about token positions in a sequence:
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
For position $pos$ and dimension $i$, different frequencies are used (from $1/10000^{0} = 1$ to $1/10000^{2/d}$). This allows the model to:
- Learn relative positions (since $\sin(\alpha + \beta) = \sin\alpha\cos\beta + \cos\alpha\sin\beta$, the encoding at position $pos+k$ can be expressed as a linear function of the encoding at $pos$).
- Attend to different frequency patterns at different layers.
- Generalize to sequence lengths longer than those seen during training.

**2. Fourier Features for Neural Radiance Fields (NeRF).** NeRF (Mildenhall et al., 2020) represents scenes as continuous 5D functions. To capture high-frequency details (textures, sharp edges), the input coordinates are mapped through Fourier features:
$$\gamma(p) = (\sin(2^0 \pi p), \cos(2^0 \pi p), \sin(2^1 \pi p), \cos(2^1 \pi p), \ldots, \sin(2^{L-1} \pi p), \cos(2^{L-1} \pi p))$$
This maps input coordinates into a higher-dimensional space using sinusoids at exponentially increasing frequencies ($2^0, 2^1, \ldots, 2^{L-1}$). The result is that the neural network can learn high-frequency functions — without these features, NeRF produces blurry results. This technique is also used in implicit neural representations and physics-informed neural networks (PINNs).

**3. SIREN Networks (Periodic Activation Functions).** Sitzmann et al. (2020) proposed SIREN (Sinusoidal Representation Networks) using $\sin$ as the activation function:
$$f(x) = \sin(Wx + b)$$
Unlike ReLU or tanh, sine activations are periodic, making them ideal for representing signals with fine detail and derivatives (e.g., images, videos, 3D shapes). The key properties:
- The derivative of $\sin$ is $\cos$, which is also periodic and bounded — avoiding vanishing/exploding gradients.
- Networks with sine activations can represent complex natural signals and their spatial/temporal derivatives.
- The $k$-th derivative of a SIREN is still a SIREN (since derivatives of sines are sines/cosines).

**4. Fourier Series in Signal Processing.** The Fourier series represents any periodic function as a sum of sines and cosines:
$$f(x) = \frac{a_0}{2} + \sum_{n=1}^\infty (a_n \cos(nx) + b_n \sin(nx))$$
This is fundamental to:
- Audio processing (compression, filtering, generation).
- Image processing (JPEG uses Discrete Cosine Transform).
- Time series analysis (spectral analysis, periodicity detection).
- Convolution theorem: convolution in time domain = multiplication in frequency domain, enabling efficient CNN implementations via FFT.

**5. Rotational Invariance in Computer Vision.** Rotation matrices use sine and cosine:
$$R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$
Capsule networks, equivariant neural networks, and group convolutional networks use trigonometric functions to build models that are rotationally invariant or equivariant.

**6. Angular Distance and Similarity.** Cosine similarity measures the cosine of the angle between vectors, used in:
- Word embeddings (word2vec, GloVe): $\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$.
- Attention mechanisms: dot-product attention computes similarity between query and key vectors.
- Dimensionality reduction: t-SNE and UMAP use angular relationships in high-dimensional spaces.

## Mathematical Explanation

**Radian Measure:** One radian is the angle subtended by an arc of length equal to the radius. A full circle is $2\pi$ radians (approximately 6.28 rad). Conversion: $\pi$ rad = $180^\circ$.

$$\theta_{\text{rad}} = \frac{\pi}{180} \theta_{\text{deg}}, \quad \theta_{\text{deg}} = \frac{180}{\pi} \theta_{\text{rad}}$$

**Unit Circle:** For angle $\theta$, the point on the unit circle is $(\cos\theta, \sin\theta)$. Key angles:

| $\theta$ (rad) | $\theta$ (deg) | $\sin\theta$ | $\cos\theta$ | $\tan\theta$ |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 0 | 0 | 1 | 0 |
| $\pi/6$ | 30 | $1/2$ | $\sqrt{3}/2$ | $1/\sqrt{3}$ |
| $\pi/4$ | 45 | $\sqrt{2}/2$ | $\sqrt{2}/2$ | 1 |
| $\pi/3$ | 60 | $\sqrt{3}/2$ | $1/2$ | $\sqrt{3}$ |
| $\pi/2$ | 90 | 1 | 0 | undefined |
| $\pi$ | 180 | 0 | $-1$ | 0 |
| $3\pi/2$ | 270 | $-1$ | 0 | undefined |
| $2\pi$ | 360 | 0 | 1 | 0 |

**Periodicity:**
- $\sin(\theta + 2\pi) = \sin\theta$, $\cos(\theta + 2\pi) = \cos\theta$
- $\tan(\theta + \pi) = \tan\theta$ (period $\pi$)

**Pythagorean Identity:**
$$\sin^2 \theta + \cos^2 \theta = 1$$
This follows from the unit circle: $x^2 + y^2 = 1$.

**Sum and Difference Formulas:**
$$\sin(A \pm B) = \sin A \cos B \pm \cos A \sin B$$
$$\cos(A \pm B) = \cos A \cos B \mp \sin A \sin B$$
$$\tan(A \pm B) = \frac{\tan A \pm \tan B}{1 \mp \tan A \tan B}$$

**Double Angle Formulas:**
$$\sin(2\theta) = 2 \sin\theta \cos\theta$$
$$\cos(2\theta) = \cos^2\theta - \sin^2\theta = 2\cos^2\theta - 1 = 1 - 2\sin^2\theta$$

**Derivatives:**
$$\frac{d}{dx} \sin x = \cos x$$
$$\frac{d}{dx} \cos x = -\sin x$$
$$\frac{d}{dx} \tan x = \sec^2 x$$

**Integrals:**
$$\int \sin x \, dx = -\cos x + C$$
$$\int \cos x \, dx = \sin x + C$$

**Euler's Formula:**
$$e^{i\theta} = \cos\theta + i\sin\theta$$
From this:
$$\cos\theta = \frac{e^{i\theta} + e^{-i\theta}}{2}, \quad \sin\theta = \frac{e^{i\theta} - e^{-i\theta}}{2i}$$

## Formula(s)

**Right triangle definitions:**
$$\sin\theta = \frac{\text{opp}}{\text{hyp}}, \quad \cos\theta = \frac{\text{adj}}{\text{hyp}}, \quad \tan\theta = \frac{\text{opp}}{\text{adj}}$$

**Unit circle:**
$$(\cos\theta, \sin\theta) = \text{point on unit circle at angle } \theta$$

**Fundamental identity:**
$$\sin^2\theta + \cos^2\theta = 1$$

**Sum formulas:**
$$\sin(A+B) = \sin A \cos B + \cos A \sin B$$
$$\cos(A+B) = \cos A \cos B - \sin A \sin B$$

**Double angle:**
$$\sin(2\theta) = 2\sin\theta\cos\theta$$
$$\cos(2\theta) = \cos^2\theta - \sin^2\theta$$

**Derivatives:**
$$\frac{d}{dx} \sin x = \cos x, \quad \frac{d}{dx} \cos x = -\sin x$$

**Euler's formula:**
$$e^{i\theta} = \cos\theta + i\sin\theta$$

**Fourier series:**
$$f(x) = \frac{a_0}{2} + \sum_{n=1}^\infty (a_n \cos(nx) + b_n \sin(nx))$$

**Positional encoding (Transformer):**
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

## Properties

1. **Periodicity:** $\sin$ and $\cos$ have period $2\pi$; $\tan$ has period $\pi$. The function values repeat at regular intervals.

2. **Boundedness:** $|\sin x| \leq 1$, $|\cos x| \leq 1$ for all real $x$. $\tan x$ is unbounded (approaches $\pm\infty$ at asymptotes).

3. **Parity:** $\sin(-x) = -\sin x$ (odd function, symmetric about origin). $\cos(-x) = \cos x$ (even function, symmetric about $y$-axis).

4. **Domain:** $\sin$ and $\cos$ have domain $\mathbb{R}$. $\tan x$ has domain $\mathbb{R} \setminus \{\frac{\pi}{2} + n\pi\}$.

5. **Range:** $\sin$ and $\cos$: $[-1, 1]$. $\tan$: $\mathbb{R}$.

6. **Continuity:** All trigonometric functions are continuous on their domains.

7. **Differentiability:** $\sin$ and $\cos$ are infinitely differentiable on $\mathbb{R}$. $\frac{d}{dx} \sin x = \cos x$, $\frac{d}{dx} \cos x = -\sin x$.

8. **Orthogonality:** For integer $m, n$: $\int_0^{2\pi} \sin(mx) \cos(nx) \, dx = 0$, $\int_0^{2\pi} \sin(mx) \sin(nx) \, dx = \pi \delta_{mn}$ (where $\delta$ is the Kronecker delta). This makes sine and cosine a basis for representing periodic functions.

9. **Phase Shift:** $\sin(x + \pi/2) = \cos x$, $\cos(x - \pi/2) = \sin x$. Sine and cosine are the same function shifted by $\pi/2$.

10. **Amplitude and Frequency:** $f(x) = A \sin(\omega x + \phi)$ has amplitude $A$, angular frequency $\omega$, period $2\pi/\omega$, and phase $\phi$.

## Step-by-Step Worked Examples

### Example 1: Evaluating Trigonometric Functions

Find $\sin(\pi/3)$, $\cos(\pi/3)$, and $\tan(\pi/3)$.

**Step 1:** Recall the unit circle. At $\theta = \pi/3 = 60^\circ$, the point is $(1/2, \sqrt{3}/2)$.

**Step 2:** $\sin(\pi/3) = y = \sqrt{3}/2 \approx 0.866$
**Step 3:** $\cos(\pi/3) = x = 1/2 = 0.5$
**Step 4:** $\tan(\pi/3) = y/x = (\sqrt{3}/2) / (1/2) = \sqrt{3} \approx 1.732$

**Answer:** $\sin(\pi/3) = \sqrt{3}/2$, $\cos(\pi/3) = 1/2$, $\tan(\pi/3) = \sqrt{3}$.

### Example 2: Solving Trigonometric Equations

Solve $\sin x = 1/2$ for $x \in [0, 2\pi)$.

**Step 1:** Identify angles where $\sin x = 1/2$. From the unit circle, $\sin(\pi/6) = 1/2$ and $\sin(5\pi/6) = 1/2$.

**Step 2:** Check domain $[0, 2\pi)$. Both $\pi/6$ and $5\pi/6$ are in this interval.

**Answer:** $x = \pi/6$ and $x = 5\pi/6$.

### Example 3: Verifying Trigonometric Identities

Verify the identity $\frac{\sin x}{1 - \cos x} = \csc x + \cot x$.

**Step 1:** Start with the left side and multiply numerator and denominator by $(1 + \cos x)$.
$$\frac{\sin x}{1 - \cos x} \cdot \frac{1 + \cos x}{1 + \cos x} = \frac{\sin x (1 + \cos x)}{1 - \cos^2 x}$$

**Step 2:** Use $\sin^2 x + \cos^2 x = 1$, so $1 - \cos^2 x = \sin^2 x$.
$$\frac{\sin x (1 + \cos x)}{\sin^2 x} = \frac{1 + \cos x}{\sin x}$$

**Step 3:** Split the fraction.
$$\frac{1}{\sin x} + \frac{\cos x}{\sin x} = \csc x + \cot x$$

**Answer:** The identity is verified.

### Example 4: Using Sum Formulas

Find $\sin(75^\circ)$ without a calculator.

**Step 1:** Express $75^\circ$ as a sum of known angles: $75^\circ = 45^\circ + 30^\circ$.

**Step 2:** Apply the sine sum formula.
$$\sin(75^\circ) = \sin(45^\circ + 30^\circ) = \sin 45^\circ \cos 30^\circ + \cos 45^\circ \sin 30^\circ$$

**Step 3:** Substitute known values.
$$\sin 45^\circ = \frac{\sqrt{2}}{2}, \quad \cos 45^\circ = \frac{\sqrt{2}}{2}, \quad \sin 30^\circ = \frac{1}{2}, \quad \cos 30^\circ = \frac{\sqrt{3}}{2}$$

**Step 4:** Compute.
$$\sin(75^\circ) = \frac{\sqrt{2}}{2} \cdot \frac{\sqrt{3}}{2} + \frac{\sqrt{2}}{2} \cdot \frac{1}{2} = \frac{\sqrt{6} + \sqrt{2}}{4}$$

**Answer:** $\sin(75^\circ) = \frac{\sqrt{6} + \sqrt{2}}{4} \approx 0.9659$

### Example 5: Finding Period and Amplitude

Find the amplitude, period, and phase shift of $f(x) = 3\sin(2x - \pi) + 1$.

**Step 1:** Write in standard form $f(x) = A \sin(\omega x + \phi) + D$.
$$f(x) = 3 \sin(2x - \pi) + 1$$

**Step 2:** Identify parameters.
- Amplitude: $A = 3$
- Angular frequency: $\omega = 2$
- Period: $T = \frac{2\pi}{\omega} = \frac{2\pi}{2} = \pi$
- Phase shift: $-\phi/\omega$. Write $2x - \pi = 2(x - \pi/2)$, so phase shift is $\pi/2$ to the right.
- Vertical shift: $D = 1$ (midline at $y = 1$)

**Answer:** Amplitude = 3, Period = $\pi$, Phase shift = $\pi/2$ right, Vertical shift = 1.

### Example 6: Derivative of a Trigonometric Function

Find $f'(x)$ for $f(x) = x^2 \sin x$.

**Step 1:** Apply the product rule: $(uv)' = u'v + uv'$.
$$u = x^2, \quad v = \sin x$$
$$u' = 2x, \quad v' = \cos x$$

**Step 2:** Compute.
$$f'(x) = 2x \cdot \sin x + x^2 \cdot \cos x = 2x \sin x + x^2 \cos x$$

**Answer:** $f'(x) = 2x \sin x + x^2 \cos x$

### Example 7: Fourier Feature Mapping

Given a 1D input $p = 0.75$ and $L = 3$, compute the Fourier feature encoding $\gamma(p)$.

**Step 1:** The encoding uses frequencies $2^0, 2^1, 2^2$ (since $L=3$, frequencies $2^0$ through $2^{L-1}$).
$$\gamma(p) = (\sin(2^0 \pi p), \cos(2^0 \pi p), \sin(2^1 \pi p), \cos(2^1 \pi p), \sin(2^2 \pi p), \cos(2^2 \pi p))$$

**Step 2:** Compute each term.
- $2^0 \pi p = \pi \cdot 0.75 = 0.75\pi$: $\sin(0.75\pi) = \sqrt{2}/2 \approx 0.707$, $\cos(0.75\pi) = -\sqrt{2}/2 \approx -0.707$
- $2^1 \pi p = 2\pi \cdot 0.75 = 1.5\pi$: $\sin(1.5\pi) = -1$, $\cos(1.5\pi) = 0$
- $2^2 \pi p = 4\pi \cdot 0.75 = 3\pi$: $\sin(3\pi) = 0$, $\cos(3\pi) = -1$

**Answer:** $\gamma(0.75) = (0.707, -0.707, -1, 0, 0, -1)$

## Visual Interpretation

**Sine Wave:**
```
y
1|    *       *       *
0|  *   *   *   *   *   *
 | *     * *     * *     *
-1|*       *       *
  |----|----|----|----|---- x
  0   pi/2  pi  3pi/2  2pi
```

The sine wave starts at 0, rises to 1 at $\pi/2$, returns to 0 at $\pi$, goes to -1 at $3\pi/2$, and returns to 0 at $2\pi$.

**Cosine Wave:**
```
y
1|*       *       *       *
0|  *   *   *   *   *   *
 | *     * *     * *     *
-1|      *       *       *
  |----|----|----|----|---- x
  0   pi/2  pi  3pi/2  2pi
```

Cosine starts at 1, decreases to 0 at $\pi/2$, goes to -1 at $\pi$, returns to 0 at $3\pi/2$, and back to 1 at $2\pi$.

**Tangent Function:**
The tangent has vertical asymptotes at $\theta = \pi/2 + n\pi$. Between asymptotes, it increases from $-\infty$ to $\infty$, crossing through 0 at $\theta = n\pi$.

**Unit Circle Visualization:**
As $\theta$ increases from 0 to $2\pi$, the point $(\cos\theta, \sin\theta)$ traces a circle. The sine is the projection onto the $y$-axis; the cosine is the projection onto the $x$-axis.

**Frequency and Amplitude:**
- $A \sin(x)$: changing $A$ stretches the wave vertically.
- $\sin(\omega x)$: changing $\omega$ compresses ($\omega > 1$) or stretches ($\omega < 1$) the wave horizontally.
- $\sin(x + \phi)$: changing $\phi$ shifts the wave left or right.

## Common Mistakes

1. **Using degrees instead of radians in calculus.** All derivative and integral formulas for trigonometric functions assume radian measure. In degrees, $\frac{d}{dx} \sin(x^\circ) = \frac{\pi}{180} \cos(x^\circ)$. Always use radians in calculus contexts.

2. **Confusing $\sin^2 x$ with $\sin(x^2)$.** $\sin^2 x = (\sin x)^2 = \sin x \cdot \sin x$, while $\sin(x^2)$ is the sine of $x^2$, a completely different function. For example, $\sin^2(\pi/2) = 1^2 = 1$, while $\sin((\pi/2)^2) = \sin(\pi^2/4) \approx \sin(2.467) \approx 0.624$.

3. **Forgetting that $\tan x$ is undefined at $x = \pi/2 + n\pi$.** Many students treat $\tan x$ as defined everywhere, forgetting the vertical asymptotes where $\cos x = 0$.

4. **Misapplying inverse trigonometric functions.** $\sin^{-1}(\sin x) = x$ is true only when $x$ is in the range of $\sin^{-1}$, which is $[-\pi/2, \pi/2]$. For $x = \pi$, $\sin^{-1}(\sin \pi) = \sin^{-1}(0) = 0 \neq \pi$.

5. **Assuming $\sin(A + B) = \sin A + \sin B$.** This is false. The correct sum formula is $\sin(A + B) = \sin A \cos B + \cos A \sin B$.

6. **Forgetting the negative sign in the derivative of cosine.** $\frac{d}{dx} \cos x = -\sin x$, not $\sin x$. This sign error is extremely common.

7. **Confusing amplitude and frequency.** In $f(x) = A \sin(\omega x)$, $A$ is the amplitude (maximum displacement from midline) and $\omega$ is the angular frequency. The period is $2\pi/\omega$. A larger $\omega$ means more oscillations per unit length.

8. **Solving $\sin \theta = c$ giving only one solution.** The equation $\sin \theta = 1/2$ has infinitely many solutions: $\theta = \pi/6 + 2\pi n$ and $\theta = 5\pi/6 + 2\pi n$ for integer $n$. Always consider the periodic nature.

9. **Thinking the range of sine and cosine includes all reals.** Both functions are bounded between -1 and 1. There is no real $x$ such that $\sin x = 2$.

10. **Overlooking positional encoding frequencies in transformers.** The encoding uses both $\sin$ and $\cos$ at each frequency to encode position. Using only $\sin$ would lose the ability to represent the full phase information.

## Interview Questions

### Beginner

1. **What are the sine and cosine functions? Give their definitions using the unit circle.**
   *Answer: On the unit circle (radius 1), for an angle $\theta$ from the positive $x$-axis, $\cos\theta$ is the $x$-coordinate and $\sin\theta$ is the $y$-coordinate of the point on the circle. They have period $2\pi$, domain $\mathbb{R}$, and range $[-1, 1]$.*

2. **What is the relationship between $\sin^2\theta$ and $\cos^2\theta$?**
   *Answer: $\sin^2\theta + \cos^2\theta = 1$ (the Pythagorean identity). This follows from the unit circle equation $x^2 + y^2 = 1$.*

3. **Evaluate $\sin(\pi/4)$, $\cos(\pi/6)$, and $\tan(\pi/4)$.**
   *Answer: $\sin(\pi/4) = \sqrt{2}/2$, $\cos(\pi/6) = \sqrt{3}/2$, $\tan(\pi/4) = 1$.*

4. **What is the period of $\sin x$ and $\cos x$?**
   *Answer: The period of both $\sin x$ and $\cos x$ is $2\pi$ radians (360 degrees). After $2\pi$, the values repeat: $\sin(x + 2\pi) = \sin x$ and $\cos(x + 2\pi) = \cos x$.*

5. **Convert $45^\circ$ to radians.**
   *Answer: $45^\circ \times \frac{\pi}{180^\circ} = \frac{\pi}{4}$ radians.*

### Intermediate

1. **What is the derivative of $\sin x$? Of $\cos x$? Why are these derivatives important for neural networks?**
   *Answer: $\frac{d}{dx} \sin x = \cos x$, $\frac{d}{dx} \cos x = -\sin x$. These derivatives are used in SIREN networks with sine activations: since the derivative of sine is cosine (also bounded and periodic), gradients remain well-behaved through many layers. The second derivative of $\sin x$ is $-\sin x$, meaning the function satisfies $f'' = -f$, a property exploited in representing oscillatory signals.*

2. **Explain how positional encoding works in the Transformer architecture. Why are sine and cosine functions used?**
   *Answer: The Transformer uses sinusoidal positional encodings: $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$ and $PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$. Sine and cosine are used because: (1) they provide a unique encoding for each position, (2) relative positions can be represented as linear combinations (since $\sin(\alpha+\beta) = \sin\alpha\cos\beta + \cos\alpha\sin\beta$), (3) different frequencies allow the model to attend to different resolution patterns, and (4) the encoding can generalize to sequence lengths beyond those seen in training.*

3. **Verify the identity $\sin^2 x = \frac{1 - \cos(2x)}{2}$.**
   *Proof: Start with the double-angle formula $\cos(2x) = 1 - 2\sin^2 x$. Rearranging: $\sin^2 x = \frac{1 - \cos(2x)}{2}$. This is power-reduction formula, useful for integrating $\sin^2 x$.*

4. **What are Fourier features in NeRF and why are they effective?**
   *Answer: Fourier features map input coordinates through sinusoids at multiple frequencies: $\gamma(p) = (\sin(2^0\pi p), \cos(2^0\pi p), \ldots, \sin(2^{L-1}\pi p), \cos(2^{L-1}\pi p))$. This mapping helps neural networks learn high-frequency functions (textures, edges) that they otherwise struggle with due to "spectral bias" — the tendency of ReLU networks to learn low-frequency functions first. The sinusoids at different frequencies provide a rich Fourier basis, enabling the network to represent sharp details in 3D scenes.*

5. **Solve $\cos 2x = \cos x$ for $x \in [0, 2\pi)$.**
   *Answer: Using $\cos 2x = 2\cos^2 x - 1$, we get $2\cos^2 x - 1 = \cos x \implies 2\cos^2 x - \cos x - 1 = 0 \implies (2\cos x + 1)(\cos x - 1) = 0$. So $\cos x = 1$ or $\cos x = -1/2$. Solutions: $\cos x = 1 \implies x = 0$; $\cos x = -1/2 \implies x = 2\pi/3, 4\pi/3$. Also check $x = 2\pi$. Solutions: $x = 0, 2\pi/3, 4\pi/3, 2\pi$.*

### Advanced

1. **Derive the derivative of $\sin x$ from the limit definition.**
   *Proof: $\frac{d}{dx} \sin x = \lim_{h \to 0} \frac{\sin(x+h) - \sin x}{h}$. Using the sum formula: $\sin(x+h) = \sin x \cos h + \cos x \sin h$. So $\frac{\sin(x+h) - \sin x}{h} = \frac{\sin x (\cos h - 1)}{h} + \frac{\cos x \sin h}{h}$. As $h \to 0$: $\frac{\cos h - 1}{h} \to 0$ and $\frac{\sin h}{h} \to 1$. Therefore $\frac{d}{dx} \sin x = \cos x$. These limits $\lim_{h\to0} \sin h / h = 1$ and $\lim_{h\to0} (\cos h - 1)/h = 0$ are proven geometrically using the unit circle and squeeze theorem.*

2. **Explain how SIREN networks (sinusoidal representation networks) work. What makes sine activations suitable for representing signals compared to ReLU?**
   *Answer: SIREN uses $\sin(\omega Wx + b)$ as its activation function, where $\omega$ is a frequency parameter. Key advantages over ReLU: (1) Sine is periodic and bounded, making it suitable for oscillatory signals. (2) The derivatives of sine are also sines/cosines, so gradient information is rich throughout the network (no dying ReLU problem). (3) Sine networks can represent fine details and high-frequency components that ReLU networks miss due to spectral bias. (4) The $k$-th derivative of a SIREN is a linear combination of sine activations, making it ideal for representing signals where higher-order derivatives matter (e.g., solving PDEs in physics-informed neural networks). The frequency parameter $\omega$ controls the wavelength of the activations and must be carefully initialized (typically $\omega = 30$) for stable training.*

3. **Prove that $\int_0^{2\pi} \sin(mx) \sin(nx) \, dx = 0$ for $m \neq n$ and $= \pi$ for $m = n$ (where $m, n$ are integers). Explain why this orthogonality is important for Fourier series and machine learning.**
   *Proof: For $m \neq n$, use the product-to-sum identity: $\sin(mx)\sin(nx) = \frac{1}{2}[\cos((m-n)x) - \cos((m+n)x)]$. Then $\int_0^{2\pi} \sin(mx)\sin(nx) dx = \frac{1}{2}\int_0^{2\pi} \cos((m-n)x) dx - \frac{1}{2}\int_0^{2\pi} \cos((m+n)x) dx = 0 - 0 = 0$ (since $\int_0^{2\pi} \cos(kx) dx = 0$ for $k \neq 0$). For $m = n$: $\int_0^{2\pi} \sin^2(mx) dx = \int_0^{2\pi} \frac{1 - \cos(2mx)}{2} dx = \frac{1}{2} \cdot 2\pi - 0 = \pi$. This orthogonality means that sine (and cosine) functions form an orthogonal basis for the space of square-integrable functions on $[0, 2\pi]$. In ML, this is relevant for: (1) Fourier features for NeRF and PINNs, which use sinusoidal bases to span the frequency domain, (2) spectral methods for solving differential equations, and (3) analyzing the "frequency bias" of neural networks — the observation that neural networks learn low frequencies first, which can be understood through the Fourier lens.*

## Practice Problems

### Easy

1. Find $\sin(\pi/6)$, $\cos(\pi/4)$, and $\tan(\pi/3)$.
2. Convert $120^\circ$ to radians and $3\pi/4$ radians to degrees.
3. Determine the amplitude and period of $f(x) = 5 \sin(3x)$.
4. Evaluate $\sin^2(\pi/3) + \cos^2(\pi/3)$.
5. Find $\cos(0)$ and $\sin(\pi/2)$.

### Medium

1. Solve $\cos x = -\frac{\sqrt{2}}{2}$ for $x \in [0, 2\pi)$.
2. Compute $\sin(105^\circ)$ using sum formulas.
3. Find $f'(x)$ for $f(x) = \sin x \cos x$.
4. Simplify $\frac{1 - \cos(2x)}{\sin(2x)}$ to a single trigonometric function.
5. A mass on a spring follows $x(t) = 0.2 \cos(4\pi t)$. Find its amplitude, frequency, and position at $t = 0.5$.

### Hard

1. Prove the identity $\frac{\sin x}{1 + \cos x} + \frac{1 + \cos x}{\sin x} = 2 \csc x$.
2. Derive the Fourier feature encoding for 2D input $(x, y)$ with $L = 2$ and explain how it enables NeRF to learn high-frequency details.
3. Show that the positional encoding in transformers allows the model to learn relative positions: prove that $PE_{pos+k}$ is a linear function of $PE_{pos}$ using trigonometric addition formulas.

## Solutions

### Easy Solutions

**1.** $\sin(\pi/6) = 1/2$, $\cos(\pi/4) = \sqrt{2}/2$, $\tan(\pi/3) = \sqrt{3}$.

**2.** $120^\circ \times \pi/180 = 2\pi/3$ rad. $3\pi/4 \times 180/\pi = 135^\circ$.

**3.** Amplitude = 5, Period = $2\pi/3$.

**4.** $\sin^2(\pi/3) + \cos^2(\pi/3) = (\sqrt{3}/2)^2 + (1/2)^2 = 3/4 + 1/4 = 1$.

**5.** $\cos(0) = 1$, $\sin(\pi/2) = 1$.

### Medium Solutions

**1.** $\cos x = -\sqrt{2}/2$. The reference angle is $\pi/4$. Cosine is negative in quadrants II and III. Solutions: $x = 3\pi/4$ and $x = 5\pi/4$.

**2.** $\sin(105^\circ) = \sin(60^\circ + 45^\circ) = \sin 60^\circ \cos 45^\circ + \cos 60^\circ \sin 45^\circ = (\sqrt{3}/2)(\sqrt{2}/2) + (1/2)(\sqrt{2}/2) = (\sqrt{6} + \sqrt{2})/4$.

**3.** $f'(x) = \cos x \cdot \cos x + \sin x \cdot (-\sin x) = \cos^2 x - \sin^2 x = \cos(2x)$.

**4.** $\frac{1 - \cos(2x)}{\sin(2x)} = \frac{2\sin^2 x}{2\sin x \cos x} = \frac{\sin x}{\cos x} = \tan x$.

**5.** Amplitude = 0.2 m, angular frequency $\omega = 4\pi$ rad/s, frequency $f = \omega/(2\pi) = 2$ Hz. $x(0.5) = 0.2 \cos(4\pi \cdot 0.5) = 0.2 \cos(2\pi) = 0.2$ m.

### Hard Solutions

**1.** Start with left side: $\frac{\sin x}{1+\cos x} + \frac{1+\cos x}{\sin x} = \frac{\sin^2 x + (1+\cos x)^2}{\sin x(1+\cos x)} = \frac{\sin^2 x + 1 + 2\cos x + \cos^2 x}{\sin x(1+\cos x)} = \frac{(\sin^2 x + \cos^2 x) + 1 + 2\cos x}{\sin x(1+\cos x)} = \frac{2 + 2\cos x}{\sin x(1+\cos x)} = \frac{2(1+\cos x)}{\sin x(1+\cos x)} = \frac{2}{\sin x} = 2\csc x$. The identity is proved.

**2.** For 2D input $(x, y)$ with $L = 2$, frequencies are $2^0 = 1$ and $2^1 = 2$:
$$\gamma(x, y) = (\sin(\pi x), \cos(\pi x), \sin(2\pi x), \cos(2\pi x), \sin(\pi y), \cos(\pi y), \sin(2\pi y), \cos(2\pi y))$$
This maps 2D coordinates to 8D features. In NeRF, this positional encoding is applied to the 3D point coordinates and 2D viewing direction before feeding them to the MLP. The multiple frequency bands allow the network to learn both low-frequency structures (overall shape, colors) and high-frequency details (textures, sharp edges). Without these features, NeRF produces blurry results because standard ReLU MLPs have a spectral bias toward low frequencies.

**3.** From the encoding definition:
$$PE_{pos+k, 2i} = \sin\left(\frac{pos+k}{10000^{2i/d}}\right) = \sin\left(\frac{pos}{10000^{2i/d}} + \frac{k}{10000^{2i/d}}\right)$$
Using $\sin(\alpha + \beta) = \sin\alpha\cos\beta + \cos\alpha\sin\beta$:
$$PE_{pos+k, 2i} = \sin\left(\frac{pos}{10000^{2i/d}}\right)\cos\left(\frac{k}{10000^{2i/d}}\right) + \cos\left(\frac{pos}{10000^{2i/d}}\right)\sin\left(\frac{k}{10000^{2i/d}}\right)$$
$$= PE_{pos, 2i} \cdot \cos\left(\frac{k}{10000^{2i/d}}\right) + PE_{pos, 2i+1} \cdot \sin\left(\frac{k}{10000^{2i/d}}\right)$$
Similarly for the cosine counterpart. Thus $PE_{pos+k}$ is a linear combination of $PE_{pos}$ with coefficients depending only on $k$, not on $pos$. This allows the transformer to learn relative position information through linear transformations, which is why absolute positional encodings based on sinusoids effectively encode relative positions.

## Related Concepts

- **Exponential Function** (MATH-050) — Euler's formula $e^{i\theta} = \cos\theta + i\sin\theta$ connects trigonometric and exponential functions.
- **Function** (MATH-044) — Trigonometric functions are periodic functions with domain $\mathbb{R}$ (for sin/cos).
- **Complex Numbers** (MATH-009) — Trigonometric functions are intimately related to complex numbers through Euler's formula.
- **Calculus** — Derivatives and integrals of trigonometric functions are essential for understanding oscillations and waves.
- **Fourier Series** — Any periodic function can be expressed as a sum of sines and cosines.

## Next Concepts

- **Inverse Trigonometric Functions** — $\arcsin x$, $\arccos x$, $\arctan x$, used for solving trigonometric equations and in angle computations.
- **Hyperbolic Functions** — $\sinh x = (e^x - e^{-x})/2$, $\cosh x = (e^x + e^{-x})/2$, related to trigonometry but with hyperbolas instead of circles.
- **Fourier Transform** — Extending Fourier series to non-periodic functions, fundamental to signal processing and modern ML (e.g., FFT convolutions).
- **Wavelets** — Multi-resolution analysis using localized wave-like functions.

## Summary

Trigonometric functions $\sin\theta$, $\cos\theta$, and $\tan\theta$ relate angles to ratios of sides in right triangles and to coordinates on the unit circle. They are periodic (period $2\pi$ for $\sin$ and $\cos$) and bounded ($[-1, 1]$ for $\sin$ and $\cos$). The fundamental identity $\sin^2\theta + \cos^2\theta = 1$ follows from the Pythagorean theorem. The derivatives $\frac{d}{dx}\sin x = \cos x$ and $\frac{d}{dx}\cos x = -\sin x$ make them essential for modeling oscillatory phenomena. In AI/ML, trigonometric functions are used for positional encoding in transformers (sinusoidal position embeddings), Fourier features for NeRF (mapping coordinates to sinusoidal frequency bands), periodic activation functions in SIREN networks, and cosine similarity for measuring vector relationships.

## Key Takeaways

- $\sin\theta$ and $\cos\theta$ are defined from the unit circle: $(\cos\theta, \sin\theta)$ is the point at angle $\theta$.
- Domain: $\mathbb{R}$ for $\sin$, $\cos$; Range: $[-1, 1]$.
- Period: $2\pi$ for $\sin$, $\cos$; $\pi$ for $\tan$.
- $\sin^2\theta + \cos^2\theta = 1$ (Pythagorean identity).
- Sum formulas: $\sin(A \pm B) = \sin A \cos B \pm \cos A \sin B$.
- Derivatives: $\frac{d}{dx} \sin x = \cos x$, $\frac{d}{dx} \cos x = -\sin x$.
- Euler's formula: $e^{i\theta} = \cos\theta + i\sin\theta$.
- Transformer positional encoding uses $\sin$ and $\cos$ at different frequencies.
- Fourier features (sinusoids at powers of 2 frequencies) enable NeRF to learn high-frequency details.
- SIREN networks use $\sin$ activations for representing signals with fine structure.
