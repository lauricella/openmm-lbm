# Interpolation stencils, step by step

> **Status:** in development for version 0.5.0. The stencils work on every platform, with one domain and with the
> domain decomposition, with both drag schemes, walls and open faces.

This page explains, from the beginning, how a particle that sits between the nodes of the lattice sees the fluid and
pushes on it. It needs no knowledge of the lattice Boltzmann method: only what a function, a sum and a polynomial
are. Every formula here is the one the plugin computes; the derivations and the proofs are in
[theory.md](../theory.md), section 9.

To choose a stencil (the short answer):

```python
from openmmlbm import LBMForce
force = LBMForce()
force.setInterpolationStencil(LBMForce.ThreePoint)   # or NearestNode (the default), Trilinear, Keys
```

| Stencil | Nodes | In one sentence |
|---|---|---|
| `NearestNode` | 1 | the particle talks to the closest node only; the force jumps when it changes cell |
| `Trilinear` | $`2\times2\times2 = 8`$ | straight lines between neighbouring nodes; cheap; the particle feels where it is in the cell |
| `ThreePoint` | $`3\times3\times3 = 27`$ | smooth, and a lone particle moves the same wherever it is in the cell |
| `Keys` | $`4\times4\times4 = 64`$ | cubic, the most accurate for smooth flows, some weights negative, the most expensive |

## 1. The problem: the fluid lives on the nodes, the particle does not

The fluid is known only on the nodes of the lattice: points $`\Delta x`$ apart along each axis (for example
$`\Delta x = 0.5`$ nm). Node $`i`$ of the x axis is at $`x_i = i\,\Delta x`$. A particle (a bead of a protein, say) is
at a position $`X`$ that is almost never on a node. Two questions:

1. **What fluid velocity does the particle see?** It needs it for the drag force
   $`\mathbf F = -\gamma m(\mathbf v - \mathbf u)`$, where $`\mathbf v`$ is the velocity of the particle and
   $`\mathbf u`$ that of the fluid around it.
2. **Where does the reaction go?** The fluid must receive $`-\mathbf F`$, otherwise momentum is not conserved. It can
   only receive it on nodes.

The **nearest node** answers both with one node: $`\mathbf u`$ is the velocity of the closest node, and the reaction
goes all to it. It is simple, but when the particle crosses the middle of a cell the node changes at once, and so do
the velocity it sees and the place where it pushes.

An **interpolation stencil** uses several nodes around the particle, each with a **weight** $`w_j`$:

```math
\mathbf u(X) = \sum_j w_j\,\mathbf u_j \qquad\text{(velocity seen by the particle)},\qquad
\text{reaction on node } j = -w_j\,\mathbf F .
```

The weights depend on where the particle is, and **the same weights are used for both**. Sections 2 and 3 give the
weights; section 4 explains why using the same weights matters.

## 2. One axis first

Everything is done axis by axis, so first a single axis.

**Step 1: the position in lattice units.** Divide by the spacing: $`s = X/\Delta x`$. With $`X = 1.15`$ nm and
$`\Delta x = 0.5`$ nm, $`s = 2.3`$: the particle is between node 2 and node 3, 30% of the way.

**Step 2: integer and fractional part.** $`i = \lfloor s\rfloor`$ (the node just below, here 2) and
$`f = s - i`$ (here 0.3), with $`0 \le f < 1`$.

**Step 3: the weights.** Each stencil has a function $`\phi(r)`$, its **kernel**, of the distance $`r`$ between the
particle and a node, measured in lattice spacings. The weight of node $`j`$ is

```math
w_j = \phi(s - j).
```

The kernel is zero far from the particle, so only a few nodes have a weight: 2, 3 or 4.

**Two rules that every stencil here obeys**, for every position of the particle:

```math
\sum_j w_j = 1, \qquad \sum_j w_j\,(j - s) = 0 .
```

The first means that a uniform flow is seen exactly (if all $`u_j = U`$ then $`u = U`$), and that the reactions add up
to exactly $`-\mathbf F`$. The second means that a flow that changes linearly in space is also seen exactly: if
$`u_j = a + b\,j`$ then $`u = a + b\,s`$, the value at the particle.

### 2.1 Trilinear: 2 nodes, straight lines

Kernel: $`\phi(r) = 1 - |r|`$ for $`|r| < 1`$, zero otherwise (a triangle).

Nodes $`i`$ and $`i + 1`$, with weights

```math
w_i = 1 - f, \qquad w_{i+1} = f,
```

so the velocity seen is the straight line between the two nodes:

```math
u(f) = (1 - f)\,u_i + f\,u_{i+1} = u_i + f\,(u_{i+1} - u_i).
```

Example: $`s = 2.3`$ gives $`w_2 = 0.7`$, $`w_3 = 0.3`$. At a node ($`f = 0`$) the weights are 1 and 0: the
trilinear stencil is then the nearest node. Its weights are never negative.

### 2.2 Three-point: 3 nodes around the nearest one

This kernel was built by Roma, Peskin and Berger (1999) for the immersed boundary method. Here the nodes are the
nearest node $`n = \mathrm{round}(s)`$ and its two neighbours, and the distance from the nearest node is
$`d = s - n`$, with $`-\tfrac12 \le d < \tfrac12`$ (for $`s = 2.3`$: $`n = 2`$, $`d = 0.3`$).

Kernel:

```math
\phi(r) = \begin{cases}
\dfrac{1 + \sqrt{1 - 3r^2}}{3}, & |r| \le \tfrac12,\\[2ex]
\dfrac{5 - 3|r| - \sqrt{1 - 3(1 - |r|)^2}}{6}, & \tfrac12 \le |r| \le \tfrac32,\\[2ex]
0, & |r| \ge \tfrac32 .
\end{cases}
```

With $`q = \sqrt{1 - 3d^2}`$ the three weights are

```math
w_{n-1} = \frac{2 - 3d - q}{6}, \qquad w_n = \frac{1 + q}{3}, \qquad w_{n+1} = \frac{2 + 3d - q}{6}.
```

These are not polynomials: they contain a square root. Where it comes from is a nice exercise. Ask for three weights
$`w_{n-1}, w_n, w_{n+1}`$ that satisfy the two rules above **and** a third one,

```math
w_{n-1}^2 + w_n^2 + w_{n+1}^2 = \tfrac12 \quad\text{for every } d .
```

The first two rules give $`w_{n+1} = w_{n-1} + d`$ and $`w_n = 1 - 2w_{n-1} - d`$; putting them into the third gives
the quadratic equation $`6w_{n-1}^2 + (6d - 4)\,w_{n-1} + \tfrac12 - 2d + 2d^2 = 0`$, whose discriminant is
$`4(1 - 3d^2)`$: the square root. The smaller root is the weight above. The third rule is what makes this stencil
special: the "self weight" $`\sum w^2`$, which decides how much a particle feels the fluid that it has set in motion
itself (section 5), is the same wherever the particle is. With the trilinear stencil it goes from $`\tfrac12`$ (middle of the cell) to 1
(on a node).

Example: $`d = 0`$ (on a node) gives $`\tfrac16, \tfrac23, \tfrac16`$; $`d = 0.25`$ gives $`0.0581, 0.6338, 0.3081`$.
The weights are never negative.

### 2.3 Keys: 4 nodes, a cubic

This is the cubic convolution of Keys (1981), the same as the Catmull–Rom spline. Nodes $`i - 1, i, i + 1, i + 2`$,
two on each side of the particle, with $`f = s - i`$ as in the trilinear case.

Kernel:

```math
\phi(r) = \begin{cases}
1 - \tfrac52 r^2 + \tfrac32 |r|^3, & |r| \le 1,\\[1ex]
2 - 4|r| + \tfrac52 r^2 - \tfrac12 |r|^3, & 1 \le |r| \le 2,\\[1ex]
0, & |r| \ge 2 .
\end{cases}
```

The four weights are cubic polynomials in $`f`$:

```math
\begin{aligned}
w_{i-1} &= \tfrac12\,(-f + 2f^2 - f^3) = -\tfrac12\,f\,(1 - f)^2,\\
w_{i}   &= \tfrac12\,(2 - 5f^2 + 3f^3),\\
w_{i+1} &= \tfrac12\,(f + 4f^2 - 3f^3),\\
w_{i+2} &= \tfrac12\,(-f^2 + f^3) = -\tfrac12\,f^2\,(1 - f).
\end{aligned}
```

Collecting the powers of $`f`$, the velocity seen is the cubic

```math
u(f) = u_i + \frac{u_{i+1} - u_{i-1}}{2}\,f + \frac{2u_{i-1} - 5u_i + 4u_{i+1} - u_{i+2}}{2}\,f^2
     + \frac{-u_{i-1} + 3u_i - 3u_{i+1} + u_{i+2}}{2}\,f^3 .
```

Besides the two rules, Keys obeys a third one, $`\sum_j w_j (j - s)^2 = 0`$: a flow that changes as a parabola is
also seen exactly. The price: the two outer weights are **negative** (between 0 and $`-\tfrac{2}{27}`$), so the
velocity seen can be outside the range of the node velocities, and the reaction on the two outer nodes points the
other way from that on the inner ones.

Example: $`f = 0`$ gives 0, 1, 0, 0 (the nearest node again); $`f = 0.25`$ gives −0.0703, 0.8672, 0.2266, −0.0234;
$`f = 0.5`$ gives −0.0625, 0.5625, 0.5625, −0.0625.

### 2.4 The weights at a glance

The weights of the nodes $`i - 1`$, $`i`$, $`i + 1`$, $`i + 2`$ (with $`i = \lfloor s\rfloor`$) as the particle moves
through the cell from node $`i`$ ($`f = 0`$) towards node $`i + 1`$:

| Stencil | $`f`$ | $`w_{i-1}`$ | $`w_i`$ | $`w_{i+1}`$ | $`w_{i+2}`$ |
|---|---|---|---|---|---|
| Trilinear | 0 | 0 | 1 | 0 | 0 |
| | 0.25 | 0 | 0.75 | 0.25 | 0 |
| | 0.5 | 0 | 0.5 | 0.5 | 0 |
| | 0.75 | 0 | 0.25 | 0.75 | 0 |
| Three-point | 0 | 0.1667 | 0.6667 | 0.1667 | 0 |
| | 0.25 | 0.0581 | 0.6338 | 0.3081 | 0 |
| | 0.5 | 0 | 0.5 | 0.5 | 0 |
| | 0.75 | 0 | 0.3081 | 0.6338 | 0.0581 |
| Keys | 0 | 0 | 1 | 0 | 0 |
| | 0.25 | −0.0703 | 0.8672 | 0.2266 | −0.0234 |
| | 0.5 | −0.0625 | 0.5625 | 0.5625 | −0.0625 |
| | 0.75 | −0.0234 | 0.2266 | 0.8672 | −0.0703 |

The three-point stencil moves with the nearest node: up to the middle of the cell it uses nodes $`i - 1`$ to
$`i + 1`$, after it nodes $`i`$ to $`i + 2`$; exactly at the middle the outer weight is zero on both sides, so the
weights do not jump.

## 3. Three dimensions: multiply the axes

In three dimensions the particle has three lattice coordinates $`s_x = X/\Delta x`$, $`s_y = Y/\Delta x`$,
$`s_z = Z/\Delta x`$. Each axis gets its own weights, exactly as in section 2, and **the weight of a node is the
product of the weights of its three coordinates**:

```math
W_{abc} = w^x_a\; w^y_b\; w^z_c ,
```

where $`a`$, $`b`$, $`c`$ number the nodes of the stencil along x, y and z. The velocity seen is the triple sum

```math
\mathbf u(\mathbf X) = \sum_a \sum_b \sum_c w^x_a\, w^y_b\, w^z_c\; \mathbf u_{a,b,c} .
```

The two rules hold in 3D too, because the sum of a product is the product of the sums: $`\sum W = 1 \cdot 1 \cdot 1`$.

### 3.1 Trilinear, 2 × 2 × 2 nodes

Nodes $`(i + a, j + b, k + c)`$ with $`a, b, c \in \{0, 1\}`$, fractional parts $`f_x, f_y, f_z`$. The eight weights and
the interpolating polynomial, written out:

```math
\begin{aligned}
\mathbf u(\mathbf X) ={}& (1-f_x)(1-f_y)(1-f_z)\,\mathbf u_{i,j,k} + f_x(1-f_y)(1-f_z)\,\mathbf u_{i+1,j,k}\\
&+ (1-f_x)f_y(1-f_z)\,\mathbf u_{i,j+1,k} + f_xf_y(1-f_z)\,\mathbf u_{i+1,j+1,k}\\
&+ (1-f_x)(1-f_y)f_z\,\mathbf u_{i,j,k+1} + f_x(1-f_y)f_z\,\mathbf u_{i+1,j,k+1}\\
&+ (1-f_x)f_yf_z\,\mathbf u_{i,j+1,k+1} + f_xf_yf_z\,\mathbf u_{i+1,j+1,k+1} .
\end{aligned}
```

Each weight is the volume of the box opposite to its node, in the cell divided by the particle.

### 3.2 Three-point, 3 × 3 × 3 nodes

Nodes $`(n_x + a, n_y + b, n_z + c)`$ around the nearest node, $`a, b, c \in \{-1, 0, 1\}`$, distances
$`d_x, d_y, d_z`$ from it. With the one-axis weights of section 2.2 written as functions,

```math
p_{-1}(d) = \frac{2 - 3d - \sqrt{1 - 3d^2}}{6},\qquad p_0(d) = \frac{1 + \sqrt{1 - 3d^2}}{3},\qquad
p_{+1}(d) = \frac{2 + 3d - \sqrt{1 - 3d^2}}{6},
```

the 27 weights are $`W_{abc} = p_a(d_x)\,p_b(d_y)\,p_c(d_z)`$ and

```math
\mathbf u(\mathbf X) = \sum_{a=-1}^{1}\sum_{b=-1}^{1}\sum_{c=-1}^{1} p_a(d_x)\,p_b(d_y)\,p_c(d_z)\;
\mathbf u_{n_x+a,\,n_y+b,\,n_z+c} .
```

For example the node $`(n_x - 1, n_y + 1, n_z)`$, across an edge from the nearest node, has the weight
$`p_{-1}(d_x)\,p_{+1}(d_y)\,p_0(d_z)`$. On a
node ($`d_x = d_y = d_z = 0`$) the weights are $`\tfrac23\cdot\tfrac23\cdot\tfrac23 = \tfrac{8}{27}`$ on the node
itself, $`\tfrac{2}{27}`$ on each of its 6 face neighbours, $`\tfrac{1}{54}`$ on each of its 12 edge neighbours and
$`\tfrac{1}{216}`$ on each of its 8 corners (they add up to 1).

### 3.3 Keys, 4 × 4 × 4 nodes

Nodes $`(i + a, j + b, k + c)`$ with $`a, b, c \in \{-1, 0, 1, 2\}`$ and fractional parts $`f_x, f_y, f_z`$. With the
cubic polynomials of section 2.3,

```math
\begin{aligned}
k_{-1}(f) &= \tfrac12(-f + 2f^2 - f^3), & k_0(f) &= \tfrac12(2 - 5f^2 + 3f^3),\\
k_{1}(f)  &= \tfrac12(f + 4f^2 - 3f^3), & k_2(f) &= \tfrac12(-f^2 + f^3),
\end{aligned}
```

the 64 weights are $`W_{abc} = k_a(f_x)\,k_b(f_y)\,k_c(f_z)`$ and

```math
\mathbf u(\mathbf X) = \sum_{a=-1}^{2}\sum_{b=-1}^{2}\sum_{c=-1}^{2} k_a(f_x)\,k_b(f_y)\,k_c(f_z)\;
\mathbf u_{i+a,\,j+b,\,k+c} ,
```

a polynomial of degree 3 in each of $`f_x`$, $`f_y`$, $`f_z`$ (tricubic). Since $`k_0`$ and $`k_1`$ are never
negative and $`k_{-1}`$ and $`k_2`$ never positive, the sign of a weight is $`(-1)^m`$, $`m`$ being the number of its
outer indices ($`-1`$ or $`2`$), or it is zero.

### 3.4 Try it

This short program computes the weights of the three stencils for one position, with the formulas above, and checks
the two rules. Change `position` and run it again.

```python
import numpy as np

dx = 0.5                                  # lattice spacing (nm)
position = np.array([1.15, 0.80, 2.05])   # particle (nm)
s = position/dx                           # lattice coordinates

def trilinear(s):
    i = np.floor(s); f = s - i
    return i, [1 - f, f]

def three_point(s):
    n = np.floor(s + 0.5); d = s - n; q = np.sqrt(1 - 3*d*d)
    return n - 1, [(2 - 3*d - q)/6, (1 + q)/3, (2 + 3*d - q)/6]

def keys(s):
    i = np.floor(s); f = s - i
    return i - 1, [(-f + 2*f**2 - f**3)/2, (2 - 5*f**2 + 3*f**3)/2, (f + 4*f**2 - 3*f**3)/2, (-f**2 + f**3)/2]

for name, rule in (('Trilinear', trilinear), ('ThreePoint', three_point), ('Keys', keys)):
    first, w = rule(s)                    # first node and weights, per axis
    w = np.array(w)                       # w[a, axis]
    W = np.einsum('a,b,c->abc', w[:, 0], w[:, 1], w[:, 2])     # W[a, b, c] = wx[a] wy[b] wz[c]
    nodes = first[None, :] + np.arange(len(w))[:, None]        # node indices, per axis
    moment = [np.sum(w[:, k]*(nodes[:, k] - s[k])) for k in range(3)]
    print('%-10s %2d nodes, first node %s, sum of the weights %.12f, first moments %s'
          % (name, W.size, first.astype(int), W.sum(), np.round(moment, 12) + 0.0))
```

Output:

```
Trilinear   8 nodes, first node [2 1 4], sum of the weights 1.000000000000, first moments [0. 0. 0.]
ThreePoint 27 nodes, first node [1 1 3], sum of the weights 1.000000000000, first moments [0. 0. 0.]
Keys       64 nodes, first node [1 0 3], sum of the weights 1.000000000000, first moments [0. 0. 0.]
```

## 4. What the plugin does with the weights

For each coupled particle, at every step:

1. it finds the nodes of the stencil and their weights, as above (after bringing the particle back into the periodic
   box);
2. it computes the velocity of the fluid at each node, momentum divided by density, $`\mathbf u_j = \mathbf j_j/\rho_j`$,
   and the velocity seen, $`\mathbf u = \sum_j W_j \mathbf u_j`$;
3. it computes the drag and random force $`\mathbf F`$ on the particle (with the centred drag, particles whose
   stencils share nodes are solved together: [theory.md](../theory.md), section 9);
4. it gives each node the reaction $`-W_j\mathbf F`$.

**Why the same weights for both directions.** With $`\sum_j W_j = 1`$ the reactions add up to $`-\mathbf F`$: the
momentum that the particle gains is exactly the momentum the fluid loses. And because "interpolate" and "spread" use
the same numbers, one is the transpose of the other; this is what keeps the balance between the friction and the
random force, so that with a fluctuating fluid and the centred drag the particles have exactly the temperature of the
fluid ([theory.md](../theory.md), section 9).

**Walls.** A solid node of the stencil is a wall at rest: it contributes zero velocity, and its share of the reaction
goes to the wall (`getWallForce()`).

**Open faces** (`setFaceBoundary()`). The stencil must not reach across an open face into the fluid of the opposite
side. Along an axis with open faces a node of the stencil beyond the face is replaced by the last node before it (the
node index is clamped to $`0 \ldots n-1`$), which takes its weight. The weights still sum to one and interpolation and
spreading still use the same numbers, so momentum and the balance of the previous paragraph are kept. What is lost is
the exact linear field in the last cell: a particle at a fraction $`f`$ of the cell between the last node and the face
sees, with the trilinear stencil, the velocity of the last node instead of the value at its position, an error of
$`f\,\Delta u`$, where $`\Delta u`$ is the change of the velocity over one cell (with $`n = 16`$ nodes between a
plate at rest and one at velocity $`U`$, up to 5.9% of $`U`$). Keep coupled particles a few nodes away from open faces;
the particles themselves stay in OpenMM's periodic box, and a particle in the cell between the last node and the face
belongs to the side of that last node.

**Short axes.** If an axis has fewer nodes than the stencil (for example 2 nodes with Keys, a quasi two-dimensional
system), the same node appears more than once in the stencil; its weights are added together.

## 5. Which stencil?

| | Nearest node | Trilinear | Three-point | Keys |
|---|---|---|---|---|
| nodes | 1 | 8 | 27 | 64 |
| uniform flow seen exactly | yes | yes | yes | yes |
| linear flow seen exactly | no | yes | yes | yes |
| parabolic flow seen exactly | no | no | no | yes |
| weights never negative | yes | yes | yes | no |
| force when the particle moves | jumps at the middle of a cell | continuous, kinks at the nodes | smooth | smooth |
| self weight $`\sum W^2`$ | 1 | $`\tfrac18`$ to 1, mean $`\tfrac{8}{27}`$ | $`\tfrac18`$ everywhere | 0.26 to 1, mean $`(57/70)^3 = 0.54`$ |

The **self weight** $`\sum_j W_j^2`$ measures how much of its own reaction a particle feels back through the fluid.
Where it changes inside a cell, the mobility of a lone particle (how fast it moves under a constant force) depends
on where it is in the cell: with the centred drag, between a node and the centre of a cell it changes by a factor of
2.1 to 4.6 with the trilinear stencil and 1.5 to 2.7 with Keys, depending on $`\tau`$ (0.62 to 3.51), and by at most
7% with the three-point stencil, which
keeps $`\sum W^2`$ constant ([theory.md](../theory.md), section 9, and [validation.md](../validation.md),
Interpolation stencils). For particles that move through the fluid the three-point stencil is the safe choice.

**Cost.** Each node of a stencil is one more sort key and one more term in every sum: Keys costs about 8 times the
trilinear stencil in the coupling. With the explicit drag on one A100, for 56320 crowded beads on $`120^3`$ nodes, a
step takes 1.6 ms with the trilinear stencil, 3.0 ms with the three-point one and 5.5 ms with Keys (the fluid alone
0.9 ms); with the centred drag 2.6, 5.7 and 12.8 ms ([validation.md](../validation.md)).

## References

- C. K. Birdsall, D. Fuss, J. Comput. Phys. 3, 494 (1969): the trilinear ("cloud in cell") weights.
- A. M. Roma, C. S. Peskin, M. J. Berger, J. Comput. Phys. 153, 509 (1999): the three-point kernel.
- R. G. Keys, IEEE Trans. Acoust. Speech Signal Process. 29, 1153 (1981): the cubic convolution kernel.
- The full list, with the derivations, is in [theory.md](../theory.md), section 9.
