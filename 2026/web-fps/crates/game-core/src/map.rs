//! Blockout map geometry: axis-aligned boxes on the ground plane plus spawn points.

use bevy_math::{Vec3, Vec3A};

/// Axis-aligned bounding box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    /// Strict overlap test: boxes that only touch do not overlap.
    pub fn overlaps(&self, other: &Aabb) -> bool {
        self.min.x < other.max.x
            && self.max.x > other.min.x
            && self.min.y < other.max.y
            && self.max.y > other.min.y
            && self.min.z < other.max.z
            && self.max.z > other.min.z
    }

    /// Slab test. Returns the distance along `dir` (unit length) to the first hit, if within `max_t`.
    pub fn ray_hit(&self, origin: Vec3, dir: Vec3, max_t: f32) -> Option<f32> {
        let (o, d) = (Vec3A::from(origin), Vec3A::from(dir));
        let inv = d.recip();
        let t1 = (Vec3A::from(self.min) - o) * inv;
        let t2 = (Vec3A::from(self.max) - o) * inv;
        let t_near = t1.min(t2).max_element();
        let t_far = t1.max(t2).min_element();
        if t_near <= t_far && t_far >= 0.0 && t_near <= max_t {
            Some(t_near.max(0.0))
        } else {
            None
        }
    }
}

/// Index into the renderer's built-in color palette.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Material {
    Floor,
    Wall,
    Cover,
    Crate,
    Accent,
}

#[derive(Clone, Debug)]
pub struct Block {
    pub aabb: Aabb,
    pub material: Material,
}

#[derive(Clone, Debug)]
pub struct Map {
    pub blocks: Vec<Block>,
    /// Feet positions.
    pub spawns: Vec<Vec3>,
}

impl Map {
    /// Distance to the first block along a ray, or `max_t` if nothing is hit.
    pub fn raycast(&self, origin: Vec3, dir: Vec3, max_t: f32) -> f32 {
        self.blocks
            .iter()
            .filter_map(|b| b.aabb.ray_hit(origin, dir, max_t))
            .fold(max_t, f32::min)
    }

    pub fn line_of_sight(&self, from: Vec3, to: Vec3) -> bool {
        let delta = to - from;
        let dist = delta.length();
        dist < 1e-4 || self.raycast(from, delta / dist, dist) >= dist
    }

    pub fn collides(&self, aabb: &Aabb) -> bool {
        self.blocks.iter().any(|b| b.aabb.overlaps(aabb))
    }

    /// The built-in arena: a 48x48 walled square with cover of three heights.
    ///
    /// Heights are chosen around the player dimensions in `sim`: 1.0 boxes can be jumped onto,
    /// 1.2 walls hide a crouching player but can be shot over standing, and the bar along the
    /// north wall can only be passed by crouching.
    pub fn arena() -> Self {
        use Material::*;
        let mut blocks = Vec::new();
        let mut add = |min: [f32; 3], max: [f32; 3], material| {
            blocks.push(Block { aabb: Aabb::new(min.into(), max.into()), material });
        };
        // A box resting on the floor, centered at (cx, cz).
        let ground = |cx: f32, cz: f32, w: f32, d: f32, h: f32| {
            ([cx - w / 2.0, 0.0, cz - d / 2.0], [cx + w / 2.0, h, cz + d / 2.0])
        };

        const H: f32 = 24.0;
        add([-H, -1.0, -H], [H, 0.0, H], Floor);
        add([-H - 1.0, 0.0, -H - 1.0], [H + 1.0, 4.0, -H], Wall);
        add([-H - 1.0, 0.0, H], [H + 1.0, 4.0, H + 1.0], Wall);
        add([-H - 1.0, 0.0, -H], [-H, 4.0, H], Wall);
        add([H, 0.0, -H], [H + 1.0, 4.0, H], Wall);

        // Central jumpable platform and tall pillars around it.
        let (a, b) = ground(0.0, 0.0, 4.0, 4.0, 1.0);
        add(a, b, Accent);
        for (x, z) in [(6.0, 0.0), (-6.0, 0.0), (0.0, 6.0), (0.0, -6.0)] {
            let (a, b) = ground(x, z, 1.5, 1.5, 4.0);
            add(a, b, Wall);
        }
        // L-shaped cover in each quadrant.
        for (sx, sz) in [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)] {
            let (a, b) = ground(10.0 * sx, 10.0 * sz, 4.0, 1.0, 2.2);
            add(a, b, Cover);
            let (a, b) = ground(10.0 * sx + 1.5 * sx, 10.0 * sz - 2.0 * sz, 1.0, 3.0, 2.2);
            add(a, b, Cover);
        }
        // Jumpable crates on the axes.
        for (x, z) in [(16.0, 0.0), (-16.0, 0.0), (0.0, 16.0), (0.0, -16.0)] {
            let (a, b) = ground(x, z, 2.0, 2.0, 1.0);
            add(a, b, Crate);
        }
        // Crouch-height walls: hide when crouched, shoot over when standing.
        for (x, z, w, d) in [(6.0, 15.0, 5.0, 0.5), (-6.0, -15.0, 5.0, 0.5), (15.0, -6.0, 0.5, 5.0), (-15.0, 6.0, 0.5, 5.0)] {
            let (a, b) = ground(x, z, w, d, 1.2);
            add(a, b, Cover);
        }
        // Duck-under bar along the north wall, with posts.
        add([-4.0, 1.3, 20.5], [4.0, 2.3, 22.5], Accent);
        add([-4.5, 0.0, 20.5], [-4.0, 2.3, 22.5], Wall);
        add([4.0, 0.0, 20.5], [4.5, 2.3, 22.5], Wall);

        let spawns = [
            (20.0, 20.0),
            (-20.0, 20.0),
            (20.0, -20.0),
            (-20.0, -20.0),
            (13.0, 0.0),
            (-13.0, 0.0),
            (0.0, 13.0),
            (0.0, -13.0),
        ]
        .into_iter()
        .map(|(x, z)| Vec3::new(x, 0.0, z))
        .collect();

        Self { blocks, spawns }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ray_hits_box_in_front_and_misses_behind() {
        let b = Aabb::new(Vec3::new(-1.0, -1.0, -6.0), Vec3::new(1.0, 1.0, -4.0));
        assert_eq!(b.ray_hit(Vec3::ZERO, Vec3::NEG_Z, 100.0), Some(4.0));
        assert_eq!(b.ray_hit(Vec3::ZERO, Vec3::Z, 100.0), None);
        assert_eq!(b.ray_hit(Vec3::ZERO, Vec3::NEG_Z, 3.0), None);
    }

    #[test]
    fn line_of_sight_is_blocked_by_walls() {
        let map = Map::arena();
        // Through the central pillar at (6, 0).
        assert!(!map.line_of_sight(Vec3::new(3.0, 1.6, 0.0), Vec3::new(9.0, 1.6, 0.0)));
        assert!(map.line_of_sight(Vec3::new(3.0, 1.6, 3.0), Vec3::new(3.0, 1.6, -3.0)));
    }
}
