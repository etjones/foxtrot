use std::f64::{EPSILON, consts::PI};

use nalgebra_glm as glm;
use glm::{DVec2, DVec3, DVec4, DMat4};

use nurbs::{AbstractCurve, AbstractSurface, NDBSplineCurve, NDBSplineSurface, SampledSurface};
use crate::{Error, mesh::Vertex};

#[derive(Debug, Clone)]
pub enum ExtrusionCurve {
    BSpline {
        curve: nurbs::BSplineCurve,
        samples: Vec<(f64, DVec3)>,
    },
    NURBS {
        curve: nurbs::NURBSCurve,
        samples: Vec<(f64, DVec3)>,
    },
    Line {
        origin: DVec3,
        dir_unit: DVec3,
    },
}

fn curve_samples<const N: usize>(curve: &NDBSplineCurve<N>) -> Vec<(f64, DVec3)>
where
    NDBSplineCurve<N>: AbstractCurve,
{
    const NUM_SAMPLES_PER_KNOT: usize = 8;
    let mut samples = Vec::new();
    for i in 0..curve.knots.len() - 1 {
        if curve.knots[i] == curve.knots[i + 1] {
            continue;
        }
        for j in 0..NUM_SAMPLES_PER_KNOT {
            let frac = (j as f64) / (NUM_SAMPLES_PER_KNOT as f64 - 1.0);
            let u = curve.knots[i] * (1.0 - frac) + curve.knots[i + 1] * frac;
            samples.push((u, curve.point(u)));
        }
    }
    samples
}

fn project_perp(p: DVec3, dir_unit: DVec3) -> DVec3 {
    p - dir_unit * p.dot(&dir_unit)
}

fn clamp_open_interval(u: f64, u_min: f64, u_max: f64) -> f64 {
    if u < u_min {
        u_min
    } else if u > u_max {
        u_max
    } else {
        u
    }
}

fn curve_min_u<const N: usize>(curve: &NDBSplineCurve<N>) -> f64 {
    curve.min_u()
}

fn curve_max_u<const N: usize>(curve: &NDBSplineCurve<N>) -> f64 {
    curve.max_u()
}

fn closest_u_initial_guess(samples: &[(f64, DVec3)], p_perp: DVec3, dir_unit: DVec3) -> f64 {
    let mut best_u = 0.0;
    let mut best_dist = std::f64::INFINITY;
    for (u, pos) in samples {
        let d = (project_perp(*pos, dir_unit) - p_perp).norm();
        if d < best_dist {
            best_dist = d;
            best_u = *u;
        }
    }
    best_u
}

fn closest_u_newton<const N: usize>(curve: &NDBSplineCurve<N>, p_perp: DVec3, dir_unit: DVec3, u0: f64) -> f64
where
    NDBSplineCurve<N>: AbstractCurve,
{
    let u_min = curve_min_u(curve);
    let u_max = curve_max_u(curve);
    let mut u_i = clamp_open_interval(u0, u_min, u_max);

    // A coarse convergence threshold is fine for visualization
    let eps1 = 0.01;
    for _ in 0..64 {
        let derivs = curve.derivs::<2>(u_i);
        let c = project_perp(derivs[0], dir_unit);
        let c_p = project_perp(derivs[1], dir_unit);
        let c_pp = project_perp(derivs[2], dir_unit);
        let r = c - p_perp;

        if r.norm() <= eps1 {
            return u_i;
        }

        let denom = c_pp.dot(&r) + c_p.norm_squared();
        if denom.abs() <= std::f64::EPSILON {
            return u_i;
        }
        let delta = -c_p.dot(&r) / denom;
        let u_next = clamp_open_interval(u_i + delta, u_min, u_max);

        if ((u_next - u_i) * c_p.norm()).abs() <= eps1 {
            return u_next;
        }
        u_i = u_next;
    }
    u_i
}

impl ExtrusionCurve {
    pub fn new_bspline(curve: nurbs::BSplineCurve) -> Self {
        let samples = curve_samples(&curve);
        Self::BSpline { curve, samples }
    }

    pub fn new_nurbs(curve: nurbs::NURBSCurve) -> Self {
        let samples = curve_samples(&curve);
        Self::NURBS { curve, samples }
    }

    pub fn new_line(origin: DVec3, dir_unit: DVec3) -> Self {
        Self::Line { origin, dir_unit }
    }

    pub fn point(&self, u: f64) -> DVec3 {
        match self {
            Self::BSpline { curve, .. } => curve.point(u),
            Self::NURBS { curve, .. } => curve.point(u),
            Self::Line { origin, dir_unit } => *origin + *dir_unit * u,
        }
    }

    pub fn deriv1(&self, u: f64) -> DVec3 {
        match self {
            Self::BSpline { curve, .. } => curve.derivs::<1>(u)[1],
            Self::NURBS { curve, .. } => curve.derivs::<1>(u)[1],
            Self::Line { dir_unit, .. } => *dir_unit,
        }
    }

    pub fn closest_u_perp(&self, p: DVec3, dir_unit: DVec3) -> f64 {
        let p_perp = project_perp(p, dir_unit);
        match self {
            Self::BSpline { curve, samples } => {
                let u0 = closest_u_initial_guess(samples, p_perp, dir_unit);
                closest_u_newton(curve, p_perp, dir_unit, u0)
            }
            Self::NURBS { curve, samples } => {
                let u0 = closest_u_initial_guess(samples, p_perp, dir_unit);
                closest_u_newton(curve, p_perp, dir_unit, u0)
            }
            Self::Line { origin, dir_unit: line_dir } => {
                // Choose u as signed distance along the line direction.
                (p - *origin).dot(line_dir)
            }
        }
    }
}

// Represents a surface in 3D space, with a function to project a 3D point
// on the surface down to a 2D space.
#[derive(Debug, Clone)]
pub enum Surface {
    Cylinder {
        location: DVec3,
        axis: DVec3,
        mat: DMat4,
        mat_i: DMat4,
        radius: f64,
        z_min: f64,
        z_max: f64,
    },
    Plane {
        normal: DVec3,
        mat_i: DMat4,
    },
    Cone {
        mat: DMat4,
        mat_i: DMat4,
        angle: f64,
    },
    BSpline(SampledSurface<3>),
    NURBS(SampledSurface<4>),
    Sphere {
        location: DVec3,
        mat: DMat4,     // uv to world
        mat_i: DMat4,   // world to uv
        radius: f64,
    },
    Torus {
        axis: DVec3,
        location: DVec3,
        mat: DMat4,
        mat_i: DMat4,
        major_radius: f64,
        minor_radius: f64,
    },
    LinearExtrusion {
        curve: ExtrusionCurve,
        dir_unit: DVec3,
    },
}

impl Surface {
    pub fn new_sphere(location: DVec3, radius: f64) -> Self {
        Surface::Sphere {
            // mat and mat_i are built in prepare()
            mat: DMat4::identity(),
            mat_i: DMat4::identity(),
            location, radius,
        }

    }

    pub fn new_linear_extrusion(curve: ExtrusionCurve, dir_unit: DVec3) -> Self {
        Surface::LinearExtrusion { curve, dir_unit }
    }
    pub fn new_cylinder(axis: DVec3, ref_direction: DVec3, location: DVec3, radius: f64) -> Self {
        let mat = Self::make_rigid_transform(axis, ref_direction, location);
        Surface::Cylinder {
            mat,
            mat_i: mat.try_inverse().expect("Could not invert"),
            axis, radius, location,
            z_min: 0.0,
            z_max: 0.0,
        }
    }

    pub fn new_torus(location: DVec3, axis: DVec3,
                     major_radius: f64, minor_radius: f64) -> Self
    {
        Surface::Torus {
            // mat and mat_i are built in prepare()
            mat: DMat4::identity(),
            mat_i: DMat4::identity(),
            location, axis, major_radius, minor_radius
        }
    }

    pub fn new_plane(axis: DVec3, ref_direction: DVec3, location: DVec3) -> Self {
        Surface::Plane {
            mat_i: Self::make_rigid_transform(axis, ref_direction, location)
                .try_inverse()
                .expect("Could not invert"),
            normal: axis,
        }
    }

    pub fn new_cone(axis: DVec3, ref_direction: DVec3, location: DVec3, angle: f64) -> Self {
        let mat = Self::make_rigid_transform(axis, ref_direction, location);
        Surface::Cone {
            mat,
            mat_i: mat.try_inverse().expect("Could not invert"),
            angle,
        }
    }

    pub fn make_affine_transform(z_world: DVec3, x_world: DVec3, y_world: DVec3, origin_world: DVec3) -> DMat4 {
        let mut mat = DMat4::identity();
        mat.set_column(0, &glm::vec3_to_vec4(&x_world));
        mat.set_column(1, &glm::vec3_to_vec4(&y_world));
        mat.set_column(2, &glm::vec3_to_vec4(&z_world));
        mat.set_column(3, &glm::vec3_to_vec4(&origin_world));
        *mat.get_mut((3, 3)).unwrap() =  1.0;
        mat
    }

    fn make_rigid_transform(z_world: DVec3, x_world: DVec3, origin_world: DVec3) -> DMat4 {
        let mut mat = DMat4::identity();
        mat.set_column(0, &glm::vec3_to_vec4(&x_world));
        mat.set_column(1, &glm::vec3_to_vec4(&z_world.cross(&x_world)));
        mat.set_column(2, &glm::vec3_to_vec4(&z_world));
        mat.set_column(3, &glm::vec3_to_vec4(&origin_world));
        *mat.get_mut((3, 3)).unwrap() =  1.0;
        mat
    }

    fn surf_lower<const N: usize>(p: DVec3, surf: &SampledSurface<N>) -> Result<DVec2, Error>
        where NDBSplineSurface<N>: AbstractSurface
    {
        surf.uv_from_point(p).ok_or(Error::CouldNotLower)
    }

    /// Lowers a 3D point on a specific surface into a 2D space defined by
    /// the surface type.  This should only be called from `lower_verts`,
    /// to ensure that `prepare` is called first.
    fn lower(&self, p: DVec3) -> Result<DVec2, Error> {
        let p_ = DVec4::new(p.x, p.y, p.z, 1.0);
        match self {
            Surface::Plane { mat_i, .. } => {
                Ok(glm::vec4_to_vec2(&(mat_i * p_)))
            },
            Surface::Cone { mat_i, .. } => {
                let xy = glm::vec4_to_vec2(&(mat_i * p_));
                Ok(DVec2::new(-xy.x, xy.y))
            },

            Surface::Cylinder { mat_i, z_min, z_max, .. } => {
                let p = mat_i * p_;
                // We convert the Z coordinates to either add or subtract from
                // the radius, so that we maintain the right topology (instead
                // of doing something like theta-z coordinates, which wrap
                // around awkwardly).

                // Scale from radius=1 to radius=0.5 based on Z
                let z = (p.z - z_min) / (z_max - z_min);
                let scale = 1.0 / (1.0 + z);
                Ok(DVec2::new(p.x * scale, p.y * scale))
            },
            Surface::Torus { mat_i, major_radius, minor_radius, .. } => {
                let p = mat_i * p_;
                /*
                         ^ Y
                         |
                    /---------\
                   /     |     \
                   |   -----   |
                   |   | O |- -|- - >Z
                   |   -----   |
                   \           /
                    \---------/

                    (X axis points into the screen)
                */
                let major_angle = p.y.atan2(p.z);

                // Rotate the point so that it's got Y = 0, so we can calculate
                // the minor angle
                let z = DVec3::new(0.0, major_angle.sin(), major_angle.cos());
                let new_mat = Self::make_rigid_transform(
                    z, DVec3::new(1.0, 0.0, 0.0), z * *major_radius);
                let new_mat_i = new_mat.try_inverse()
                    .expect("Could not invert");
                let new_p = new_mat_i * DVec4::new(p.x, p.y, p.z, 1.0);

                let minor_angle = new_p.x.atan2(new_p.z);

                // Construct nested circles with a scale based on the ratio
                // of radiuses (to make an _attempt_ to match 3D distance)
                let scale = 1.0 + (major_radius / minor_radius) *
                                  (major_angle + PI) / (2.0 * PI);

                let x = if *major_radius > 0.0 {
                    -minor_angle.cos()
                } else {
                    minor_angle.cos()
                };
                Ok(scale * DVec2::new(x, minor_angle.sin()))
            },
            Surface::BSpline(surf) => Self::surf_lower(p, surf),
            Surface::NURBS(surf) => Self::surf_lower(p, surf),
            Surface::Sphere { mat_i, radius, .. } => {
                // mat_i is constructed in prepare to be a reasonable basis
                let p = (mat_i * p_).xyz() / *radius;
                let r = p.yz().norm();

                // Angle from 0 to PI
                let angle = r.atan2(p.x);
                let yz = p.yz();
                Ok(if yz.norm() < EPSILON {
                    yz
                } else {
                    yz * angle / yz.norm()
                })
            },
            Surface::LinearExtrusion { curve, dir_unit } => {
                let u = curve.closest_u_perp(p, *dir_unit);
                let c = curve.point(u);
                let v = (p - c).dot(dir_unit);
                Ok(DVec2::new(u, v))
            },
        }
    }

    fn prepare(&mut self, verts: &[Vertex]) {
        match self {
            Surface::Cylinder { mat_i, z_min, z_max, .. } => {
                *z_min = std::f64::INFINITY;
                *z_max = -std::f64::INFINITY;
                for v in verts {
                    let p = (*mat_i) * DVec4::new(v.pos.x, v.pos.y, v.pos.z, 1.0);
                    if p.z < *z_min {
                        *z_min = p.z;
                    }
                    if p.z > *z_max {
                        *z_max = p.z;
                    }
                }
            },
            Surface::Sphere { mat, mat_i, location, .. } => {
                let ref_direction = (verts[0].pos - *location).normalize();
                let d1 = (verts.last().unwrap().pos - *location).normalize();
                let axis = ref_direction.cross(&d1).normalize();

                *mat = Self::make_rigid_transform(
                        axis, ref_direction, *location);
                *mat_i = mat
                    .try_inverse()
                    .expect("Could not invert");
            },
            Surface::Torus { axis, mat, mat_i, location, .. } => {
                let mean_dir = verts.iter()
                    .map(|v| v.pos - *location)
                    .sum::<DVec3>()
                    .normalize();
                let mean_perp_dir = (mean_dir - *axis * mean_dir.dot(axis)).normalize();
                *mat = Self::make_rigid_transform(
                    mean_perp_dir, *axis, *location);
                *mat_i = mat
                    .try_inverse()
                    .expect("Could not invert");
            },
            _ => (),
        }
    }

    pub fn lower_verts(&mut self, verts: &mut [Vertex])
        -> Result<Vec<(f64, f64)>, Error>
    {
        self.prepare(verts);
        let mut pts = Vec::with_capacity(verts.len());
        for v in verts {
            // Project to the 2D subspace for triangulation
            let proj = self.lower(v.pos)?;
            // Update the surface normal
            v.norm = self.normal(v.pos, proj);
            pts.push((proj.x, proj.y));
        }
        // If this is a BSpline surface, calculate an aspect ratio based on the
        // control points net, then use it to transform projected points.  This
        // means that positions in 2D (UV) space are closer to positions in 3D
        // space, so the triangulation is better.
        let aspect_ratio = match self {
            Surface::NURBS(surf) => Some(surf.surf.aspect_ratio()),
            Surface::BSpline(surf) => Some(surf.surf.aspect_ratio()),
            _ => None,
        };
        if let Some(aspect_ratio) = aspect_ratio {
            for p in pts.iter_mut() {
                p.1 *= aspect_ratio;
            }
        }
        Ok(pts)
    }

    pub fn raise(&self, uv: DVec2) -> Option<DVec3> {
        match self {
            Surface::Sphere { mat, radius, .. } => {
                let angle = uv.norm();
                if angle > PI {
                    return None;
                }
                let x = angle.cos();

                // Calculate pre-transformed position
                let pos = (*radius) * if uv.norm() < EPSILON {
                    DVec3::new(x, 0.0, 0.0)
                } else {
                    let yz = uv.normalize() * angle.sin();
                    DVec3::new(x, yz.x, yz.y)
                };
                // Transform into world space
                let pos = (mat * DVec4::new(pos.x, pos.y, pos.z, 1.0))
                    .xyz();
                Some(pos)
            },
            Surface::BSpline(s) => Some(s.surf.point(uv)),
            Surface::NURBS(s) => Some(s.surf.point(uv)),
            Surface::Torus { mat, minor_radius, major_radius, .. } => {
                let mut uv = uv;
                if *major_radius > 0.0 {
                    uv.x *= -1.0;
                }
                let minor_angle = uv.y.atan2(uv.x);
                let major_angle = (uv.norm() - 1.0) /
                                  (major_radius / minor_radius) * 2.0 * PI - PI;
                let new_p = DVec3::new(minor_angle.sin(), 0.0, minor_angle.cos()) * *minor_radius;

                let z = DVec3::new(0.0, major_angle.sin(), major_angle.cos());
                let new_mat = Self::make_rigid_transform(
                    z, DVec3::new(1.0, 0.0, 0.0), z * *major_radius);
                let p = new_mat * DVec4::new(new_p.x, new_p.y, new_p.z, 1.0);

                Some((mat * p).xyz())
            },
            _ => unimplemented!(),
        }
    }

    fn bbox(pts: &[(f64, f64)]) -> (f64, f64, f64, f64) {
        let (mut xmin, mut xmax) = (std::f64::INFINITY, -std::f64::INFINITY);
        let (mut ymin, mut ymax) = (std::f64::INFINITY, -std::f64::INFINITY);
        for (px, py) in pts {
            xmin = px.min(xmin);
            ymin = py.min(ymin);
            xmax = px.max(xmax);
            ymax = py.max(ymax);
        }
        (xmin, xmax, ymin, ymax)
    }

    pub fn add_steiner_points(&self, pts: &mut Vec<(f64, f64)>,
                                     verts: &mut Vec<Vertex>)
    {
        let (xmin, xmax, ymin, ymax) = Self::bbox(&pts);
        let num_pts = match self {
            Surface::Sphere { .. }   => 6,
            Surface::Torus { .. } => 32,
            _ => 0,
        };

        for x in 0..num_pts {
            let x_frac = (x as f64 + 1.0) / (num_pts as f64 + 1.0);
            let u = x_frac * xmax + (1.0 - x_frac) * xmin;
            for y in 0..num_pts {
                let y_frac = (y as f64 + 1.0) / (num_pts as f64 + 1.0);
                let v = y_frac * ymax + (1.0 - y_frac) * ymin;

                let uv = DVec2::new(u, v);
                if let Some(pos) = self.raise(uv) {
                    pts.push((u, v));
                    verts.push(Vertex {
                        pos,
                        norm: self.normal(pos, uv),
                        color: DVec3::new(0.0, 0.0, 0.0),
                    });
                }
            }
        }
    }

    fn surf_normal<const N: usize>(uv: DVec2, surf: &SampledSurface<N>) -> DVec3
        where NDBSplineSurface<N>: AbstractSurface
    {
        // Calculate first order derivs, then cross them to get normal
        let derivs = surf.surf.derivs::<1>(uv);
        let n = derivs[1][0].cross(&derivs[0][1]);
        n.normalize()
    }

    // Calculate the surface normal, using either the 3D or 2D position
    pub fn normal(&self, p: DVec3, uv: DVec2) -> DVec3 {
        match self {
            Surface::Plane { normal, .. } => *normal,
            Surface::Cone { mat, mat_i, angle, .. } => {
                // Project into CONE SPACE
                let pos = mat_i * DVec4::new(p.x, p.y, p.z, 1.0);
                let xy = if pos.xy().norm() > std::f64::EPSILON {
                    pos.xy().normalize()
                } else {
                    return DVec3::zeros();
                };
                let normal = DVec4::new(xy.x * angle.cos(),
                                        xy.y * angle.cos(), -angle.sin(), 0.0);
                // Deproject back into world space
                (mat * normal).xyz()
            }
            Surface::Sphere { location, .. } => (p - location).normalize(),
            Surface::Cylinder { mat, mat_i, .. } => {
                // Project the point onto the axis
                let proj = mat_i * DVec4::new(p.x, p.y, p.z, 1.0);

                // Then the normal is just pointing along that direction
                // (same hack as below)
                let norm = DVec3::new(proj.x, proj.y, 0.0).normalize();
                (mat * norm.to_homogeneous()).xyz()
            },
            Surface::BSpline(surf) => Self::surf_normal(uv, surf),
            Surface::NURBS(surf) => Self::surf_normal(uv, surf),
            Surface::Torus { mat, mat_i, major_radius, .. } => {
                let p = (*mat_i * DVec4::new(p.x, p.y, p.z, 1.0)).xyz();
                let major_angle = p.y.atan2(p.z);

                let z = DVec3::new(0.0, major_angle.sin(), major_angle.cos()) * *major_radius;
                let norm = (p - z).normalize();

                (mat * norm.to_homogeneous()).xyz()
            },
            Surface::LinearExtrusion { curve, dir_unit } => {
                let du = curve.deriv1(uv.x);
                let n = du.cross(dir_unit);
                if n.norm() <= std::f64::EPSILON {
                    // Degenerate tangent; fall back to something stable-ish
                    (p - (p.dot(dir_unit) * *dir_unit)).normalize()
                } else {
                    n.normalize()
                }
            },
        }
    }
}
