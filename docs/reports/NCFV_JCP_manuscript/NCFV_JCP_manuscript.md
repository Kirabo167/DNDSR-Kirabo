---
author:
- Author names and affiliations to be inserted
bibliography: references.bib
csl: numeric.csl
date: 15 September 2026
reference-section-title: References
title: A differential-integration node-centred finite-volume framework
  for efficient high-order reconstruction on unstructured meshes
---

# Abstract {#abstract .unnumbered}

High-order node-centred finite-volume methods incur substantial costs
from quadratic coefficient storage and repeated nonlinear flux
evaluation. Building on the author's master's thesis, this article
develops a differential-integration formulation for unstructured dual
meshes. A complete quadratic least-squares problem supplies second-order
gradients, while exact simplex moments and total-differential identities
replace Hessian contractions by precomputed gradient weights. The
resulting operators recover nodal point values from cell averages and
integrate physical fluxes without production Gauss points. A chain-rule
error analysis establishes the required flux-gradient accuracy under
smoothness and bounded-stencil assumptions. A frozen macroface
dissipation matrix reduces the Riemann work to one call per primal edge,
and an edge-owner MPI protocol shares one conservative flux. The thesis
cost model predicts 39.8% fewer floating-point storage entries and a
10.31-fold reduction in multiplications for a representative tetrahedral
stencil. Its reported one-dimensional and extruded-vortex tests show
near-third-order convergence. Independently available DNDSR paired runs
yield marching-time ratios of 4.94--17.36 and aggregate
proportional-set-size reductions of 15.6--66.5%. The repository's
present component-wise vortex convergence sequence does not yet
establish stable third-order accuracy. The formulation therefore
identifies both the measured efficiency gains and the remaining accuracy
conditions associated with nonplanar dissipation, viscous closure and
boundary integration.

**Keywords:** high-order finite volume; node-centred discretization;
unstructured mesh; quadrature-free integration; $k$-exact
reconstruction; MPI

# 1. Introduction {#sec:intro}

High-order methods remain attractive for vortex-dominated and
wave-propagation flows because reduced numerical dissipation can lower
the resolution required for a prescribed error. Their use on
unstructured meshes is nevertheless limited by reconstruction, flux
integration and data-management costs [@wang2007review]. The classical
$k$-exact finite-volume route reconstructs a polynomial from cell
averages and evaluates interface fluxes by sufficiently accurate
quadrature [@barth1990; @ollivier2002]. The reconstruction stencil, the
number of face quadrature points and the cost of nonlinear Riemann
evaluations all increase with order and dimension.

Several research directions address parts of this cost. Quadrature-free
ADER finite-volume schemes use analytic integration and a local
space--time procedure [@dumbser2007]; spectral-volume formulations have
reconstructed a polynomial flux and analytically integrated it over
subfaces [@wang2002sv; @harris2008]. Compact and multiple-correction
methods instead seek to control stencil growth or reuse local
differential information [@pont2017; @wang2017]. Vertex-centred
median-dual formulations are of particular interest because their edge
loop can reduce the number of control volumes and flux interfaces, and
their high-order $k$-exact variants have been incorporated in production
solvers [@setzwein2021; @setzwein2022]. These approaches establish that
high-order reconstruction, analytic integration and edge-based data
layouts are complementary rather than competing design choices.

The immediate methodological basis is Ma's master's thesis on an
efficient third-order vertex-centred finite-volume method
[@ma2026thesis], especially Chapters 3--5. It combines gradient-only
reconstruction, differential integration and a macroface dissipation
approximation. The present article develops that formulation in a
unified notation, makes its consistency assumptions explicit, and
relates its analytical cost model and reported tests to the
independently available DNDSR implementation evidence. The underlying
gradient-only method is therefore attributed to the thesis.

Two antecedents clarify this construction. Liu and Vinokur [@liu1998]
derive polynomial moments on segments, triangles and tetrahedra,
supplying the geometric integration identities. Nishikawa and White
[@nishikawa2023] eliminate stored second derivatives in a third-order
cell-centred tetrahedral scheme using nodal gradients and projected
derivatives. Their solution unknowns are point values at cell centres.
Here the evolved unknowns are dual-cell averages associated with primal
vertices: the anchor generally differs from the integration-simplex
centroid, so first moments and the full quadratic moment contraction
must be retained. This difference explains the separate point-recovery
operator and the general weights developed below.

The central observation is that a complete quadratic reconstruction
determines a consistent first derivative, while its second-derivative
coefficients need not be stored or evaluated at run time. For each
micro-simplex of a dual control volume or dual interface, the Hessian
contribution can be written as a precomputed linear combination of
gradients at its construction vertices. The resulting formulation has
the following components:

1.  a mixed-element node-centred dual geometry whose internal macrofaces
    correspond one-to-one with primal edges;

2.  a differential-integration identity that replaces quadratic
    coefficient storage and production Gauss-point storage with gradient
    weights;

3.  a macroface flux formulation and edge-owner MPI protocol that
    preserve a unique numerical flux across partition boundaries; and

4.  a paired comparison against a conventional quadratic-plus-quadrature
    path, reporting both performance/memory measurements and the present
    limits of the accuracy evidence.

The aim is not to relabel every analytic integral as "quadrature free."
In this paper the term means that the *production* differential path
creates, stores and traverses no volume, internal-face or boundary-face
quadrature points. High-order quadrature is still used as an independent
diagnostic and as the conventional baseline. This distinction is
essential when numerical accuracy and resource use are discussed
together.

# 2. Node-centred finite-volume formulation {#sec:method}

## 2.1 Dual control volumes and semi-discrete system

Let $\Omega_i$ be the dual control volume associated with primal node
$i$, let $V_i=|\Omega_i|$, and let $e=(i,j)$ denote a primal edge. The
dual macroface $\Gamma_e$ separates the two endpoint control volumes. In
two dimensions, each cell--edge--node chain produces a micro-triangle
formed by the node, edge midpoint and arithmetic mean of cell vertices.
In three dimensions, each cell--face--edge--node chain similarly
produces a micro-tetrahedron using the node, edge midpoint, arithmetic
face point and arithmetic cell point. These are definitions of the dual
geometry; for non-simplex primal cells the arithmetic points are not
substituted by area or volume centroids.

For the inviscid conservation law
$\partial_t\bm{U}+\bm{\nabla}\cdot\bm{F}(\bm{U})=\bm{0}$, the
node-centred semi-discrete system is
$$\frac{\mathrm{d}\overline{\bm{U}}_i}{\mathrm{d}t}
 =-\frac{1}{V_i}\left(\sum_{e\in\mathcal{E}(i)}B_{ie}\widehat{\bm\Phi}_e
 +\widehat{\bm\Phi}^{\partial}_i\right),
 
\;(1)$$ where $\overline{\bm{U}}_i$ is a dual-cell mean,
$B_{ie}\in\{-1,+1\}$ is the oriented node--edge incidence,
$\widehat{\bm\Phi}_e$ is the area-integrated numerical flux, and
$\widehat{\bm\Phi}^{\partial}_i$ is the boundary contribution. The same
topology is used by the differential and conventional paths; only the
integration and storage representations differ.

## 2.2 Quadratic reconstruction with gradient-only runtime storage {#sec:recon}

For a scalar conservative component, write the mean-preserving
polynomial as
$$p_i(\bm x)=\bar u_i+\sum_{\ell=1}^{N_q}a_{i\ell}\psi_{i\ell}(\bm x),
 \text{ }
 \psi_{i\ell}=b_\ell(\bm\xi)-\frac{1}{V_i}\int_{\Omega_i}b_\ell(\bm\xi)\,\mathrm{d}V,
 \text{ } N_q=\frac{d(d+3)}{2}.
 
\;(2)$$ Thus $\int_{\Omega_i}p_i\,\mathrm{d}V=V_i\bar u_i$ for any
coefficient vector. This is the zero-mean construction of thesis Section
3.2 [@ma2026thesis]. Its moment subtraction is essential because a
dual-cell average generally differs from the value at its anchor vertex.

At each node, a weighted least-squares problem is constructed from
neighbouring dual-cell means in a complete quadratic basis. With local
scaled coordinate $\bm\xi=\bm H_i^{-1}(\bm x-\bm x_i)$,
$\bm H_i=\operatorname{diag}(\Delta x_i,\Delta y_i,\Delta z_i)$ contains
the directional half-spans of the *full dual control volume*. The
half-spans are accumulated from micro-simplex construction vertices;
they are not taken from a primal-cell bounding box or a
reconstruction-stencil bounding box. On translated periodic boundaries,
relative-coordinate lower/upper bounds are first united across periodic
copies, which prevents the periodic-box width from contaminating a local
scale.

Let $\bm b_{im}$ be the difference between the mean quadratic basis in
neighbour control volume $m$ and the target-control-volume mean basis. A
weighted SVD supplies $$\bm a_i=\bm P_i
 \left[
  \overline{\bm{U}}_{m_1}-\overline{\bm{U}}_{i},\ldots,
  \overline{\bm{U}}_{m_s}-\overline{\bm{U}}_{i}
 \right]^{T},

\;(3)$$ More explicitly, with
$(\bm A_i)_{m\ell}=V_m^{-1}\int_{\Omega_m}\psi_{i\ell}\,\mathrm{d}V$ and
positive diagonal weight matrix $\bm W_i$, the retained map is
$$\bm P_i=(\bm W_i^{1/2}\bm A_i)^\dagger\bm W_i^{1/2},
 \text{ }
 \bm P_i^{(g)}=[\,\bm I_d\ \bm 0\,]\bm P_i.
 
\;(4)$$ The thesis writes the unweighted full-rank map as
$(\bm A_i^T\bm A_i)^{-1}\bm A_i^T$; (4) expresses the same least-squares
selection with the weighting and SVD used by the implementation. The
quadratic columns participate before the derivative rows are selected.
Dropping those columns before inversion would change the least-squares
problem.

The thesis begins with face-neighbour dual cells, equivalent to
primal-edge neighbours, and expands through successive neighbour layers
when necessary. Its suggested three-dimensional stencil size of 14--18
for nine unknown coefficients is a practical starting point. Stencil
cardinality alone does not establish unisolvency: the rank and
conditioning tests must also pass, particularly near walls and on
anisotropic grids. where $\bm P_i$ is accepted only if the complete
quadratic system has full numerical rank and its condition number is
below the specified threshold. The nonconstant basis has five rows in
two dimensions and nine rows in three dimensions; its pure-square
entries retain the $\xi_\alpha^2/2$ convention. Thus the least-squares
solve is complete-quadratic in both modes. The differential path stores
only the first $d$ rows of $\bm P_i$, which yield the physical gradient
$$\bm G_i=\bm{\nabla}\bm{U}_i=\bm H_i^{-1}\bm a_{1,i}.
 
\;(5)$$ Thus, the method does not replace a quadratic reconstruction by
a linear one; it removes only the unnecessary runtime representation of
its second-derivative block. The reference-length matrix is used
consistently when evaluating the target and neighbour mean bases and
when restoring physical gradients. The scalar $V_i^{1/d}$ remains
available for distance weights and CFL/wall scales but is not
substituted for $\bm H_i$ in the polynomial basis.

The dual-cell average and point value are related by a precomputed
stencil,
$$\overline{\bm{U}}_i=\bm{U}_i+\sum_{m\in\mathcal{S}_i}\bm w_{im}\cdot\bm G_m,
 \text{ }
 \bm{U}_i=\overline{\bm{U}}_i-\sum_{m\in\mathcal{S}_i}\bm w_{im}\cdot\bm G_m.
 
\;(6)$$ Here and below a vector weight acts independently on every
conservative component. Equation (6) is also the recovery operation used
before constructing interface states.

## 2.3 Offline compression of simplex moments {#sec:differential}

Consider a line, triangle or tetrahedral micro-simplex $K$ with $n$
construction vertices $\bm r_r$, expanded about anchor node $i$. Define
$\bm d_r=\bm r_r-\bm x_i$ and $\bm D=\sum_{r=1}^{n}\bm d_r$. Every
construction point is an affine combination of original primal nodes, so
its gradient is interpolated by the same coefficients,
$$\bm g(\bm r_r)=\sum_m C_{rm}\bm g_m.

\;(7)$$ The exact first and second moments about that anchor are
$$\int_K\bm d\,\mathrm{d}\mu=\frac{|K|}{n}\bm D,\text{ }
 \int_K\bm d\bm d^T\,\mathrm{d}\mu=
 \frac{|K|}{n(n+1)}
 \left(\bm D\bm D^T+\sum_{r=1}^{n}\bm d_r\bm d_r^T\right).
 
\;(8)$$ Here $\mathrm{d}\mu$ is the intrinsic measure of $K$, and
$n=2,3,4$ for a segment, triangle and tetrahedron. These are the moment
identities used in thesis Eqs. (3-12)--(3-15) [@ma2026thesis; @liu1998].
They retain $\bm D\bm D^T$ even when the anchor is outside a surface
simplex. Only a centroid-based expansion has $\bm D=\bm 0$. For a
quadratic component $u$, the total differential gives
$$\bm{\mathcal H}^{(u)}_i\bm d_r=\bm g(\bm r_r)-\bm g_i+\mathcal{O}(h^2),
 
\;(9)$$ where $\bm{\mathcal H}^{(u)}_i$ is the Hessian and
$\bm g=\bm{\nabla}u$. Substitution into the exact moments of a quadratic
polynomial produces
$$\int_K u\,\mathrm{d}V=|K|u_i+\sum_m\bm\ell^{K}_{im}\cdot\bm g_m .
 
\;(10)$$ In the implemented affine representation, the weight is
explicitly $$\begin{gathered}
 \bm\ell^{K}_{im}={}\frac{|K|}{n}\bm D\,\delta_{im}\\
 +\frac{|K|}{2n(n+1)}
 \left[
   \bm D\sum_{r=1}^{n}(C_{rm}-\delta_{im})
  +\sum_{r=1}^{n}\bm d_r(C_{rm}-\delta_{im})
 \right].
\end{gathered}

\;(11)$$ Equation (11) is evaluated once during geometry initialization
and its nonzero contributions are merged by support-node index. It has
the same form for a volume micro-simplex and a surface micro-simplex,
with $|K|$ interpreted as the appropriate line, area or volume measure.
The control-volume weights, divided by $V_i$, give the $\bm w_{im}$ in
(6); therefore point recovery is a direct compressed integral, not an
extrapolation from sampled Gauss values.

No volume, internal-face or boundary-face Gauss point is created after
this compression. The conventional path, in contrast, retains all
quadratic coefficients and evaluates them at fourth-order quadrature
locations. The distinction is material: the efficient method removes
both the location arrays and the repeated polynomial/flux evaluations
that a point loop would require.

## 2.4 Polynomial exactness and local accuracy of point recovery {#sec:local-accuracy}

The mechanism of thesis Section 3.2.4 [@ma2026thesis] can be stated with
explicit error bounds. Suppose a shape-regular family of micro-simplices
has diameter $O(h)$, the interpolation reproduces affine functions, and
the stencil coefficients remain uniformly bounded. For a quadratic $u$,
its gradient is affine, so exact nodal gradients give
$\bm g(\bm r_r)-\bm g_i=\bm{\mathcal H}^{(u)}\bm d_r$ exactly.
Substitution into (8) proves quadratic exactness of (10). This argument
includes off-centre anchors.

For a smooth $u\in C^3$, define the differential functional
$Q_{K,i}(u_i,\{\bm g_m\})=|K|u_i+\sum_m\bm\ell^K_{im}\cdot\bm g_m$. The
Taylor remainder and the moment-weight scaling give
$$\left|Q_{K,i}(u_i,\{\bm g_m\})-\int_Ku\,\mathrm{d}\mu\right|
 \le C|K|h^3,\text{ }
 \sum_m\|\bm\ell^K_{im}\|\le C|K|h.
 
\;(12)$$ Consequently, using numerical gradients
$\bm g_{m,h}=\bm g_m+O(h^2)$ adds at most $O(|K|h^3)$ to the integral.
Affine interpolation at an edge midpoint, arithmetic face point or
arithmetic cell point retains this $O(h^2)$ gradient accuracy, provided
all participating distances are $O(h)$.

After summing micro-volumes and dividing by $V_i$, point recovery
satisfies $$|u_{i,h}-u(\bm x_i)|
 \le |\bar u_{i,h}-\bar u_i|+Ch^3.
 
\;(13)$$ Thus second-order gradients suffice for third-order point
recovery from sufficiently accurate averages. This is a local
reconstruction statement: evolved cell averages carry their own spatial
and temporal errors. A face-integral error $O(h^{d+2})$, divided by a
cell volume $O(h^d)$, only supplies an $O(h^2)$ residual bound without
further cancellation. Stability and inter-face error structure must
therefore be addressed before inferring third-order solution convergence
from polynomial exactness alone.

## 2.5 Nonlinear flux mapping and chain-rule accuracy {#sec:chain-rule}

Thesis Section 4.1 [@ma2026thesis] provides the step linking
reconstruction to physical-flux integration. Let
$\bm{U}_h-\bm{U}=O(h^p)$ and
$\bm G_{\alpha,h}-\partial_\alpha\bm{U}=O(h^q)$, where
$\bm G_{\alpha,h}$ denotes the component vector for derivative direction
$\alpha$. Let $\bm A_\beta(\bm{U})=\partial F_\beta/\partial\bm{U}$ be
bounded and Lipschitz on the range of states. Then $$\begin{gathered}
 \bm E_{\alpha\beta}={}\bm A_\beta(\bm{U}_h)\bm G_{\alpha,h}
       -\partial_\alpha F_\beta(\bm{U}),\\
 \bm E_{\alpha\beta}={}[\bm A_\beta(\bm{U}_h)-\bm A_\beta(\bm{U})]\bm G_{\alpha,h}
       +\bm A_\beta(\bm{U})(\bm G_{\alpha,h}-\partial_\alpha\bm{U})
       =O(h^{\min(p,q)}).
\end{gathered}

\;(14)$$ Also $F_\beta(\bm{U}_h)-F_\beta(\bm{U})=O(h^p)$. With
$(p,q)=(3,2)$, the physical-flux value is third-order accurate and its
chain-rule gradient is second-order accurate, precisely the inputs
required by (12). For compressible Euler flow these statements apply to
smooth states with density and pressure bounded away from zero. They do
not apply across a shock or after a limiter activates. No nonzero
derivative assumption is required for this upper bound; a vanishing
Jacobian component may improve the local error.

## 2.6 Macroface state, physical-flux and dissipation assembly {#sec:macroface}

Let $S_e$ be the scalar sum of micro-surface measures and $\bm A_e$ the
oriented vector sum. On a nonplanar macroface, $S_e$ and $\|\bm A_e\|$
are deliberately kept distinct. For an interior macroface, the
side-averaged states are
$$\overline{\bm{U}}_{e,L}=\bm{U}_i+\frac{1}{S_e}\sum_m\bm\omega^{L}_{em}\cdot\bm G_m,
 \text{ }
 \overline{\bm{U}}_{e,R}=\bm{U}_j+\frac{1}{S_e}\sum_m\bm\omega^{R}_{em}\cdot\bm G_m,
 
\;(15)$$ where $\bm\omega_{em}^{L/R}$ are accumulated versions of (11).
The corresponding physical-flux integral for side $s\in\{L,R\}$ is
$$\bm\Phi^c_{e,s}=
 \bm F(\bm{U}_{i_s})\cdot\bm A_e+
 \phi_{i_s}\sum_m\sum_{\alpha,\beta}
 W^{s}_{em,\alpha\beta}\,
 \partial_\alpha F_\beta(\bm{U}_m),
 
\;(16)$$ where $W^{s}_{em,\alpha\beta}$ is the precomputed flux-gradient
weight, $\phi_{i_s}$ is the side limiter factor and
$\partial_\alpha F_\beta(\bm{U}_m)$ is calculated from the already
available state gradient. The implementation caches these physical-flux
gradients for owned and ghost support nodes once per residual
evaluation; this replaces repeated chain-rule work in every edge loop.

The single Riemann call per macroface is used only for the nonlinear
dissipative correction: $$\widehat{\bm\Phi}_e =
 \frac{1}{2}\left(\bm\Phi^c_{e,L}+\bm\Phi^c_{e,R}\right)
 +S_e\left[
 \bm F^{\mathrm{num}}(\overline{\bm{U}}_{e,L},\overline{\bm{U}}_{e,R};\bm n_e)
 -\frac{1}{2}\left(
 \bm F(\overline{\bm{U}}_{e,L})\cdot\bm n_e+
 \bm F(\overline{\bm{U}}_{e,R})\cdot\bm n_e\right)
 \right],
 
\;(17)$$ with $\bm n_e=\bm A_e/\|\bm A_e\|$. Equation (17) prevents the
central physical flux from being approximated by a single state
evaluation while avoiding a Riemann solve at every micro-surface point.
Roe's approximate solver [@roe1981] is one available flux option; the
actual flux selected in a verification case must always be reported.

For the Roe--Pike form used in thesis Section 4.2.1 [@ma2026thesis],
define
$\bm B_e=|\widetilde{\bm A}_e|=\bm R_e|\bm\Lambda_e|\bm R_e^{-1}$, with
any entropy correction included in $|\bm\Lambda_e|$. The correction in
(17) becomes $$\bm\Phi^d_e=-\frac{S_e}{2}\bm B_e
       (\overline{\bm{U}}_{e,R}-\overline{\bm{U}}_{e,L}).
 
\;(18)$$ The Roe velocity and enthalpy are formed from the two macroface
means, e.g.
$\widetilde{\bm v}=(\sqrt{\rho_L}\bm v_L+\sqrt{\rho_R}\bm v_R)/
(\sqrt{\rho_L}+\sqrt{\rho_R})$, with the analogous formula for
$\widetilde H$. Freezing this matrix over the macroface permits it to be
taken outside the jump integral. This is the algebraic reason that one
characteristic decomposition replaces the repeated decompositions at
quadrature points.

The approximation deserves a geometry qualification. On a planar smooth
face, if $\bm{U}_R-\bm{U}_L=O(h^3)$ and $\|\bm B(\bm x)-\bm B_e\|=O(h)$,
the change in the integrated dissipative term is $O(S_eh^4)$. On a
nonplanar macroface, microface normals may differ by $O(1)$ under
refinement, so the second assumption does not follow from state
smoothness. The thesis's single effective normal is retained as the
discrete model, while its order on general mixed grids is assessed by
full-scheme tests. This prevents the small-jump argument from being used
as an unconditional global-order proof.

Each macroface owns one sorted, de-duplicated `efficientStencil`. A
stencil entry contains left/right state-gradient weights, left/right
flux-gradient matrices and the scalar weight used for surface-gradient
recovery. Zero-order state and flux contributions are not copied into
every entry: they are represented once through the two anchor nodes,
$S_e$ and $\bm A_e$. This structure makes the inner loop a contiguous
traversal of synchronized point values, state gradients and cached
physical-flux gradients rather than a search through several sparse
arrays.

## 2.7 Limiter, boundary treatment and residual dataflow {#sec:runtime}

For shock control, the efficient limiter constrains the macroface
averages in (15), not a primal-edge midpoint value. Bounds are
constructed from endpoint point values $\bm{U}_i$ and $\bm{U}_j$, and a
single factor $\phi_i$ multiplies the *whole* gradient correction of
side $i$. Scaling every contributing support-node gradient by that
node's factor would define a different interface mean and is not used.
Point recovery itself remains unlimited; the factor is applied to
interface averages, physical-flux-gradient integrals and viscous surface
gradients.

For a scalar component, write the unlimited mean correction as
$\delta_{ie}=\bar u_{e,i}-u_i$, and let $u^{\min}_{ij}=\min(u_i,u_j)$
and $u^{\max}_{ij}=\max(u_i,u_j)$. An explicit statement of the
endpoint-bounded limiter discussed in thesis Section 3.3 [@ma2026thesis]
is $$\phi_{ie}=
 \begin{cases}
  \min(1,(u^{\max}_{ij}-u_i)/\delta_{ie}),&\delta_{ie}>0,\\
  \min(1,(u^{\min}_{ij}-u_i)/\delta_{ie}),&\delta_{ie}<0,\\
  1,&\delta_{ie}=0,
 \end{cases}
 \text{ } \phi_i=\min_{e\in\mathcal E(i)}\phi_{ie}.
 
\;(19)$$ The bounded mean is $u_i+\phi_i\delta_{ie}$; (15) displays the
unlimited case. A system implementation must specify how component
factors and positivity protection are combined. Endpoint bounds can
activate even in smooth non-monotone regions, so formal smooth-order
arguments assume an inactive limiter.

Boundary micro-surfaces use a compact affine stencil with a scalar
integration weight. The inviscid boundary flux is evaluated at each
support-node state by the boundary flux function and then integrated by
the affine weights. This ordering matters for nonlinear boundary
conditions: interpolating states first and applying the boundary flux
once would be a different discrete operator. When viscous terms are
enabled, the same boundary stencil yields the averaged interior
gradient; the current inviscid efficiency claims do not imply a verified
third-order viscous scheme.

One efficient residual evaluation executes the following dataflow:

1.  pull dual-cell means; apply the precomputed first-derivative rows of
    (3); pull gradients;

2.  recover nodal point values with (6); pull point values and, when
    enabled, limiter factors;

3.  compute physical-flux gradients once per local/ghost support node;

4.  on each owned edge, evaluate (16) on both sides, form (17), and
    compute the edge spectral-radius contribution;

5.  pull ghost-edge fluxes and spectral radii; assemble the signed
    node--edge divergence in (1); then apply the periodic quotient
    average where relevant.

Only node means, gradients, point values, limiter factors and edge
flux/radius fields participate in runtime communication. Micro-geometry
and Gauss point fields do not.

## 2.8 Viscous extension and affine boundary integration {#sec:viscous-extension}

Thesis Section 4.2.2 [@ma2026thesis] proposes a face-averaged viscous
closure. With our convention
$\bm G\in\mathbb R^{d\times n_{\mathrm{var}}}$, its gradient correction
can be written
$$\bm G_e^{\mathrm{corr}}=\frac{\overline{\bm G}_{e,L}+\overline{\bm G}_{e,R}}{2}
  +\frac{\bm n_e(\overline{\bm{U}}_{e,R}-\overline{\bm{U}}_{e,L})^T}{2\tilde h_e},
 \text{ } \tilde h_e=\frac{\min(V_i,V_j)}{S_e}.
 
\;(20)$$ The normal-jump contribution couples the two states in addition
to averaging gradients. For primitive variables $\bm q=\bm q(\bm{U})$, a
chain rule maps $\bm G_e^{\mathrm{corr}}$ to
$\bm G_{q,e}^{\mathrm{corr}}$, and the proposed integral has the form
$$\bm\Phi^{v}_e\simeq
 S_e\bm F^v\!\left(
       \frac{\bar{\bm q}_{e,L}+\bar{\bm q}_{e,R}}2,\bm G_{q,e}^{\mathrm{corr}},\bm n_e\right).
 
\;(21)$$ These formulas document the thesis extension; the paired
runtime results below concern the inviscid implementation. A
second-order gradient does not provide a general third-order viscous
residual. In addition, moving a nonlinear constitutive law outside an
integral is an approximation: temperature-dependent transport
coefficients and products of variable velocity and gradients require a
separate consistency analysis.

For a planar boundary triangle $T$ with three construction vertices and
an affine numerical-flux interpolant $f_b$, the correctly normalized
integral is
$$\int_T f_b\,\mathrm{d}S=\frac{|T|}{3}\sum_{r=1}^3 f_b(\bm r_r),
 \text{ }
 f_b(\bm r_r)\simeq\sum_m C_{rm}f_b(\bm{U}_m;\bm n_T).
 
\;(22)$$ This expands the boundary construction of thesis Section 4.3.2
[@ma2026thesis]. The $1/3$ factor is required by constant preservation;
summing nodal values with $|T|$ alone would triple a constant flux. The
nodal flux includes the boundary-state construction before affine
interpolation. Curved boundaries also require normal and
geometric-approximation control. Concentrating a lower-order error on
the boundary is insufficient on its own to guarantee third-order global
accuracy: the PDE, stability, boundary closure and selected norm all
matter.

::: {#tab:representation}
  Property                              Conventional quadrature   Differential integration          Consequence
  ------------------------------------ ------------------------- -------------------------- ----------------------------
  2-D/3-D stored reconstruction rows             5 / 9                     2 / 3               smaller dynamic state
  Volume and surface points              stored and traversed         0 in production             no point arrays
  State at an integration location       quadratic evaluation       gradient functional          no Hessian storage
  Riemann calls on one macroface           per surface point                one              fewer nonlinear flux calls
  Partitioned macroface                    local point loop          unique edge owner            one shared flux

  : Table 1. Runtime representation of the two implementations. The
  reconstruction solve is quadratic in both paths; "rows" are
  nonconstant reconstruction rows per solution component.
:::

## 2.9 MPI ownership and conservation {#sec:mpi}

Each primal edge has one deterministic owner, selected from the
partition ownership of its adjacent cells. The owner evaluates
$\widehat{\bm\Phi}_e$ once; ranks holding ghost endpoints pull that
result. Node owners then apply the signed incidence in (1). The design
ensures that the two endpoints of a cross-partition edge use exactly the
same flux rather than independently recomputing rounded variants. Point
values and gradients are synchronized by the existing node pull maps.
The implementation does not retain raw pointers into communication
arrays across allocation changes.

# 3. Cost model and amortization of geometry preprocessing {#sec:cost}

The storage and arithmetic model in thesis Section 5.1 [@ma2026thesis],
printed pp. 57--59, makes the sources of efficiency quantitative. It
assumes a pure tetrahedral primal grid, five conservative variables, an
inactive limiter and inviscid fluxes. Let $s$ be the mean least-squares
stencil size, $N_f$ the mean number of adjacent dual cells, and $N_e$
the mean number of primal tetrahedra around one edge. Each dual
macroface then contains $2N_e$ micro-triangles. The thesis baseline uses
three quadrature points per micro-triangle. Contributions stored or
computed once on a shared face are charged one half to each endpoint.

Under that counting convention, the numbers of floating-point storage
entries per dual cell are $$M_T=16N_fN_e+14s+99,\text{ }
 M_E=6N_fN_e+20N_f+8s+44.
 
\;(23)$$ The corresponding numbers of multiplications per spatial
residual evaluation are $$\begin{gathered}
 C_T=621N_fN_e+45s,\\
 C_E=22.5N_fN_e+168.5N_f+15s+84.
 \end{gathered}
 
\;(24)$$ These reproduce thesis Tables 5.1--5.3. The reduction from
$45s$ to $15s$ reflects retaining three instead of nine coefficients for
five equations. The conventional term proportional to $N_fN_e$ includes
state evaluations, nonlinear fluxes and entropy corrections at microface
quadrature points. The efficient path substitutes cached nodal flux
gradients and support-weight contractions, with dissipation computed
once per macroface. The thesis calls this a per-time-step count; here it
is interpreted as the spatial work per residual evaluation, to make the
extra evaluations of a multistage integrator explicit.

::: {#tab:cost-model}
  Quantity                           Conventional   Differential     Interpretation
  -------------------------------- -------------- -------------- ------------------
  Floating-point storage entries             1269            764   39.80% reduction
  Multiplications per residual              37935           3681    $C_T/C_E=10.31$

  : Table 2. Thesis cost model evaluated at $s=15$, $N_f=12$, $N_e=5$.
  Counts are per dual cell; the ratios are recalculated from thesis
  Table 5.3 [@ma2026thesis].
:::

The model predicts arithmetic work and algorithmic arrays, rather than
elapsed time or process memory. It excludes additions, divisions, square
roots, cache behaviour, index storage, communication buffers and
allocator overhead. Moreover, the DNDSR baseline below uses six surface
quadrature points per micro-triangle and its current fused stencil
layout, so the thesis counts cannot be inserted as measured DNDSR bytes
or timings.

For a fixed mesh, let $P_E,P_T$ be preprocessing times and $c_E,c_T$
measured costs of one residual evaluation. A run with $N_R$ residual
evaluations satisfies the simple model
$$T_E=P_E+N_Rc_E,\text{ } T_T=P_T+N_Rc_T.
 
\;(25)$$ If $P_E>P_T$ and $c_T>c_E$, the break-even count is
$N_R>(P_E-P_T)/(c_T-c_E)$. If $P_E\le P_T$ and $c_E<c_T$, no positive
break-even delay occurs in this model. This extension of the thesis
count explains why a marching-only speed ratio should be accompanied by
preprocessing cost for short runs or moving meshes.

At fixed polynomial order and bounded stencil sizes, both paths retain
linear asymptotic work and storage in the number of nodes and edges. The
gain is a reduction in constants. Runtime communication can become a
larger fraction of the faster edge loop, so a serial arithmetic
reduction does not by itself predict the parallel speedup.

# 4. Implementation and verification protocol {#sec:implementation}

The algorithms were implemented as two selectable modes in the DNDSR
NCFV module: `EfficientDifferential` and `TraditionalQuadrature`. They
share mesh preprocessing, least-squares stencil construction, limiters,
time integration, flux family and MPI ownership. The conventional mode
retains all quadratic coefficients and uses fourth-order rules (14
points per micro-tetrahedron and 6 per micro-triangle in the reported
three-dimensional setting). The differential mode retains the gradient
block and geometry/flux weights only.

The verification evidence used here consists of: (i) micro-geometry
closure, (ii) polynomial-reproduction audits with independently
evaluated reference quadrature, (iii) owner/ghost and periodic
consistency checks, (iv) static reconstruction on a translated
isentropic vortex, and (v) completed, paired time-marching tests.
Production tests retain no differential-mode Gauss points; diagnostic
quadrature is isolated from production storage. Geometry closure errors
are approximately $10^{-16}$ in the reported periodic meshes, and
parallel reproductions agree to roundoff in the corresponding audits.

![Figure 1. Static first-reconstruction errors of the differential mode
for a periodic isentropic vortex. The left panel measures recovered
nodal values and the right panel measures first derivatives. This is a
$t=0$ reconstruction check, not an end-to-end convergence
demonstration.](convergence.png){#fig:static-reconstruction
width="0.98\\linewidth"}

# 5. Numerical evidence {#sec:results}

## 5.1 Accuracy and application results reported in the thesis {#sec:thesis-results}

The thesis contains additional completed tests beyond the repository
evidence available for the previous manuscript draft. The values in
Tables 3 and 4 are transcribed from its numbered Tables 5.5 and 5.6
(printed pp. 61--62; PDF pages 75--76) [@ma2026thesis]. They are
source-reported results, not newly executed simulations. The nominal
convergence rate is $p_{\mathrm{obs}}=\log(E_h/E_{h/2})/\log 2$; the
rounded source errors reproduce the stated rates to the displayed
precision.

::: {#tab:thesis-1d}
    $N$   Advection--diffusion error   Rate         Burgers error   Rate
  ----- ---------------------------- ------ --------------------- ------
     50          $2.94\times10^{-2}$     --   $7.41\times10^{-4}$     --
    100          $3.73\times10^{-3}$   2.98   $9.72\times10^{-5}$   2.93
    200          $4.58\times10^{-4}$   3.02   $1.37\times10^{-5}$   2.83
    400          $5.70\times10^{-5}$   3.01   $1.65\times10^{-6}$   3.06
    800          $7.11\times10^{-6}$   3.00   $2.10\times10^{-7}$   2.97

  : Table 3. One-dimensional $L_1$ errors reported in thesis Table 5.5
  [@ma2026thesis]. Advection--diffusion is advanced for 10.5 periods,
  and Burgers' equation to $t=0.1$.
:::

::: {#tab:thesis-vortex}
    Nominal $h$   Control volumes   Reported $L_1$ error   Rate
  ------------- ----------------- ---------------------- ------
              1               745    $3.36\times10^{-3}$     --
          $1/2$              4752    $3.89\times10^{-4}$   3.11
          $1/4$             33694    $4.74\times10^{-5}$   3.04
          $1/8$            252780    $5.64\times10^{-6}$   3.07

  : Table 4. Extruded isentropic-vortex results reported in thesis Table
  5.6 [@ma2026thesis]: periodic domain $[0,10]\times[0,10]\times[0,4]$,
  $t=2$. The source labels its metric as an $L_1$ error, without
  specifying the component in the table heading.
:::

Figure 2 redraws these tabulated values. The one-dimensional and
extruded-vortex series support the thesis's reported near-third-order
behaviour for those configurations. The extruded vortex has no imposed
spanwise variation, so it exercises a three-dimensional mesh and
operator without testing a fully three-dimensional solution field. Its
$L_1$ metric and nominal mesh size differ from the component-wise $L_2$
metric and measured $h$ used by the current repository sequence.
Although control-volume counts coincide for the prism sequence, equality
of counts does not establish equality of flux, time-step, initialization
or error-evaluation settings.

![Figure 2. Convergence redrawn from thesis Tables 5.5--5.6
[@ma2026thesis]. Dashed lines are third-order guides anchored at the
finest value of each dataset; they are not additional simulation
results. No repository error series is mixed into these source-reported
curves.](thesis_convergence.png){#fig:thesis-convergence
width="\\linewidth"}

Thesis Sections 5.2.3--5.2.5 also report three application studies. The
two-dimensional Riemann problems use an extruded $400\times400\times5$
mesh, testing interacting shocks and contact discontinuities. The
laminar flat-plate test uses $M_\infty=0.2$, $Re_L=10^5$, $Pr=0.72$, an
adiabatic wall, a $200\times100\times5$ grid and pseudo-time CFL 30; its
velocity and skin-friction curves are compared with Blasius profiles.
These plots support the reported application behaviour, but neither
supplies a viscous grid-convergence study.

For the CRM wing--body, the thesis reports approximately six million
mesh entities and a 512-core calculation at $M_\infty=0.85$ and
$\alpha=2^\circ$, with $C_L=0.4513$ and $C_D=0.0197$. The reference
values quoted by the thesis are 0.47 and 0.029. Its narrative describes
an inviscid calculation, also lists $Re=5\times10^6$ and an adiabatic
wall, and attributes the drag deficit to omitted turbulence modelling.
Those settings require clarification before a controlled validation
claim can be made. We retain this as a source-reported large-mesh
demonstration; it supplies neither new experimental data nor a measured
parallel-scaling curve.

## 5.2 Current repository accuracy evidence and its scope {#sec:accuracy}

Figure 1 shows that recovered nodal values decrease faster than third
order in the $L_2$ norm on the two finest static reconstruction levels,
while first derivatives converge close to second order, as expected from
a complete quadratic reconstruction. This does not itself prove the
order of the full transient scheme. In particular, point recovery
combines the average and gradient errors through (6).

Table 5 reports the available end-to-end periodic isentropic-vortex
calculation. Both modes use $\mathrm{CFL}=0.5$, SSPRK3, no limiter, no
viscosity and the same physical initial state; each solution reaches
$t=2$. The efficient calculation retains no Gauss points. The
conventional calculation is more accurate on these four levels, but both
series exhibit an initially low observed order and the efficient method
reaches only 2.43 for density and 2.68 for pressure in the finest $L_2$
comparison. These data establish decreasing error and a useful
controlled comparison, but they do *not* establish stable third-order
convergence of the complete method. A publication-quality order claim
requires temporal refinement/extrapolation, additional sufficiently fine
grids, and verification with the intended standard Roe rather than a
similarly named scalar-dissipation option.

::: {#tab:vortex-accuracy}
  Mesh     $h$               $\rho_E$                        $p_E$                     $\rho_T$                  $p_T$
  ------ -------- ------------------------------ ------------------------------ ----------------------- -----------------------
  iv10    0.9210      $5.1562\times10^{-2}$          $7.5891\times10^{-2}$       $4.8963\times10^{-2}$   $7.3065\times10^{-2}$
  iv20    0.4683   $2.6700\times10^{-2}$ (0.97)   $4.3078\times10^{-2}$ (0.84)   $2.2378\times10^{-2}$   $3.6804\times10^{-2}$
  iv40    0.2360   $5.7776\times10^{-3}$ (2.23)   $9.8230\times10^{-3}$ (2.16)   $4.5929\times10^{-3}$   $7.5905\times10^{-3}$
  iv80    0.1186   $1.0810\times10^{-3}$ (2.43)   $1.5524\times10^{-3}$ (2.68)   $8.3044\times10^{-4}$   $1.1824\times10^{-3}$

  : Table 5. Completed periodic-vortex results at $t=2$ on prism meshes.
  $E$ and $T$ denote differential and conventional quadrature modes,
  respectively; entries are dual-volume-weighted point-value $L_2$
  errors. The observed order in parentheses is for the differential
  path.
:::

## 5.3 Paired performance and memory measurements {#sec:performance}

Table 6 compares actual marches to $t=2$ for prism, hexahedral and
tetrahedral mesh families. Each row is paired: both algorithms use the
same mesh, MPI rank count and one thread per rank. The reported time is
the marching phase only, excluding initialization and file output; PSS
is the aggregate proportional set size at the final state. Rank counts
change across mesh levels and therefore this table is *not* a
strong-scaling study. It is a per-row algorithm comparison.

The conventional-to-differential marching-time ratio is 4.94--17.36.
Final PSS reductions are 15.6--66.5%, with the largest reductions on
tetrahedral meshes, where the conventional micro-simplex quadrature
representation is especially numerous. These measurements are consistent
with the representation changes in Table 1; they should not be
extrapolated to viscous, limited, GPU or differently partitioned runs
without direct measurement. Full hardware, compiler, allocator and
affinity metadata should accompany any submission version of this table.

::: {#tab:performance}
  Family   Mesh     ranks     nodes   $t_E$ (s)   $t_T$ (s)   $t_T/t_E$   PSS reduction
  -------- ------ ------- --------- ----------- ----------- ----------- ---------------
  Family   Mesh     ranks     nodes   $t_E$ (s)   $t_T$ (s)   $t_T/t_E$   PSS reduction
  Prism6   iv10         1       745       0.743       7.772       10.47           15.6%
  Prism6   iv20         1     4,752      11.752     122.166       10.40           44.1%
  Prism6   iv40         8    33,694      29.943     280.974        9.38           48.7%
  Prism6   iv80        32   252,780     176.211   1,260.767        7.15           42.3%
  Hex8     iv10         1     1,331       1.422      10.703        7.53           16.7%
  Hex8     iv20         1     9,261      23.578     190.532        8.08           41.2%
  Hex8     iv40         8    68,921      56.384     384.079        6.81           42.9%
  Hex8     iv80        32   531,441     352.152   1,738.351        4.94           34.7%
  Tet4     iv10         1     1,331       1.825      31.681       17.36           40.8%
  Tet4     iv20         1     9,261      32.385     540.866       16.70           66.5%
  Tet4     iv40         8    68,921      76.646   1,123.901       14.66           65.5%
  Tet4     iv80        32   531,441     412.866   4,866.953       11.79           55.3%

  : Table 6. Paired performance and memory measurements for full marches
  to $t=2$. $t_E$ and $t_T$ are differential and conventional marching
  times; PSS is aggregated across ranks.
:::

# 6. Limitations and required completion studies {#sec:limitations}

The resource measurements and the absence of production quadrature
storage are verified properties of the present implementation. The
following items remain before this draft can support a broad journal
claim:

1.  Reconcile the thesis Tables 5.5--5.6 with archived meshes, solver
    revision, flux, CFL/time step, error component and norm definition.
    The thesis reports near-third-order results; the present repository
    sequence measures a different error and does not yet reproduce an
    equivalent convergence test.

2.  Demonstrate asymptotic solution accuracy with a time-refinement
    study and more refined meshes; report $L_1$, $L_2$ and $L_\infty$
    norms separately.

3.  Repeat inviscid studies with the intended characteristic Roe flux
    and document all entropy fixes and dissipation choices. The current
    option called `Roe_M2` follows a scalar local Lax--Friedrichs-type
    dissipation path and must not be described as standard Roe.

4.  Verify viscous accuracy independently. The present differential
    treatment of inviscid volume/surface functionals does not by itself
    prove a third-order Navier--Stokes discretization.

5.  Add shock, boundary, curved-geometry and weak/strong-scaling
    studies, with compiler, CPU/GPU, memory and MPI-affinity details
    sufficient for reproduction.

These limitations are not cosmetic qualifications. They distinguish the
measured reduction of quadrature work from a claim about the asymptotic
accuracy of every configuration. This is also consistent with the
broader high-order unstructured-grid literature, where boundary
treatment, nonlinear dissipation and reconstruction stencil quality must
be validated jointly with formal polynomial consistency
[@wang2007review; @chen2022; @setzwein2021].

# 7. Conclusions {#sec:conclusion}

Ma's thesis supplies the mathematical and algorithmic basis for
gradient-only reconstruction and differential integration
[@ma2026thesis]. The expanded derivation shows how zero-mean least
squares, exact simplex moments and an $O(h^2)$ gradient approximation
give a local $O(h^3)$ point-recovery error when cell averages are
sufficiently accurate. Smooth nonlinear flux maps preserve the required
value and derivative orders. A frozen macroface Roe matrix explains the
reduced characteristic work, with separate qualifications for nonplanar
geometry, limiting and viscous terms.

A node-centred differential-integration finite-volume framework has been
formulated and implemented for unstructured mixed-element meshes. A full
quadratic least-squares reconstruction supplies gradients, while
total-differential identities turn volume and macroface integrals into
precomputed gradient weights. The production method stores no Gauss
points and no quadratic reconstruction coefficients, and it uses one
Riemann solve per primal-edge-associated macroface. An edge-owner MPI
protocol makes that flux unique across partitions.

The conventional-to-differential paired measurements show
4.94--17.36$\times$ faster marching and 15.6--66.5% lower aggregate PSS
in the current inviscid periodic tests. Those efficiency results are
substantial and directly traceable to the data representation. In
contrast, the current $t=2$ convergence sequence shows decreasing errors
without stable third-order asymptotics. The appropriate conclusion is
therefore that the framework is an efficient, well-audited candidate for
a third-order scheme, not that all accuracy requirements have already
been met. The completion studies in Section 6 provide a concrete route
from this manuscript draft to a submission-ready JCP article.

The thesis's separate $L_1$ datasets report rates near three, including
3.11, 3.04 and 3.07 for the extruded vortex. Its analytical count
predicts 39.8% lower storage and a multiplication ratio of 10.31 for
representative tetrahedral connectivity. These source results strengthen
the method's motivation and evidence base, while reproduction under
matched settings remains necessary to explain the difference from the
current repository convergence sequence.

# Data availability {#data-availability .unnumbered}

The thesis source is Runzhi Ma, *Research on the Efficient Third-order
Vertex-centered Finite Volume Method*, Tsinghua University, April 2026,
supplied by the author as a 94-page PDF. Its extracted numerical values
are distributed here as `thesis_reported_convergence.csv`; the
supplemental figure is a redraw from those values. Original solver
outputs for the thesis tests were not supplied with the PDF. The
accompanying source map records printed-page and PDF-page locations. The
algorithm implementation, case configurations, diagnostics and numerical
summaries used in this draft are maintained in the DNDSR repository. The
paired performance CSV is located at the repository root; the accuracy
summaries are under `docs/reports/ncfv_t2_cfl05_thesis_20260907` and
`docs/reports/ncfv_traditional_t2_cfl05_20260908`. A public archival
identifier and immutable revision should be added before submission.

# Declaration of competing interest {#declaration-of-competing-interest .unnumbered}

The authors declare no competing financial interests or personal
relationships that could have appeared to influence the work reported in
this paper. Replace this placeholder with the authors' approved
declaration before submission.

# CRediT authorship contribution statement {#credit-authorship-contribution-statement .unnumbered}

To be completed by the authors before submission.

# References {#bibliography .unnumbered}
