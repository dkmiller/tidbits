//! Fixed-tick simulation: movement, collision, the hitscan gun, health, and respawn.
//!
//! Humans and bots are indistinguishable here: each fighter is driven only by the [`InputCmd`]
//! handed to [`World::step`] for that tick.

use bevy_math::{EulerRot, Quat, Vec2, Vec3};

use crate::map::{Aabb, Map};
use crate::rng::Rng;

pub const TICK_HZ: f64 = 60.0;
pub const DT: f32 = (1.0 / TICK_HZ) as f32;

pub const HALF_WIDTH: f32 = 0.3;
pub const STAND_HEIGHT: f32 = 1.8;
pub const CROUCH_HEIGHT: f32 = 1.1;
pub const STAND_EYE: f32 = 1.6;
pub const CROUCH_EYE: f32 = 0.95;

const WALK_SPEED: f32 = 6.0;
const SPRINT_SPEED: f32 = 9.0;
const CROUCH_SPEED: f32 = 3.0;
const AIR_CONTROL: f32 = 4.0;
const GRAVITY: f32 = 20.0;
const JUMP_SPEED: f32 = 7.0;
const MAX_FALL_SPEED: f32 = 30.0;
const KILL_PLANE: f32 = -20.0;

pub const MAX_HEALTH: i32 = 100;
const RESPAWN_SECS: f32 = 3.0;

/// The one gun.
pub const GUN_DAMAGE: i32 = 20;
pub const GUN_INTERVAL: f32 = 0.12;
pub const GUN_RANGE: f32 = 120.0;

pub type FighterId = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Team {
    Humans,
    Bots,
}

/// Everything a controller (keyboard, bot AI, later the network) can tell a fighter to do in one tick.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct InputCmd {
    /// x = strafe right, y = forward; each in [-1, 1].
    pub movement: Vec2,
    /// Absolute view angles in radians. Yaw 0 looks down -Z; positive pitch looks up.
    pub yaw: f32,
    pub pitch: f32,
    pub jump: bool,
    pub crouch: bool,
    pub sprint: bool,
    pub fire: bool,
}

#[derive(Clone, Debug)]
pub struct Fighter {
    pub id: FighterId,
    pub name: String,
    pub team: Team,
    /// Feet position, and its value at the start of the last tick (for render interpolation).
    pub pos: Vec3,
    pub prev_pos: Vec3,
    pub vel: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub on_ground: bool,
    pub crouching: bool,
    pub health: i32,
    /// Seconds until respawn; only meaningful while dead.
    pub respawn_in: f32,
    pub fire_cooldown: f32,
    pub kills: u32,
    pub deaths: u32,
}

impl Fighter {
    pub fn alive(&self) -> bool {
        self.health > 0
    }

    pub fn height(&self) -> f32 {
        if self.crouching { CROUCH_HEIGHT } else { STAND_HEIGHT }
    }

    pub fn eye(&self) -> Vec3 {
        self.pos + Vec3::Y * if self.crouching { CROUCH_EYE } else { STAND_EYE }
    }

    pub fn aabb(&self) -> Aabb {
        body_aabb(self.pos, self.height())
    }

    pub fn look_dir(&self) -> Vec3 {
        look_dir(self.yaw, self.pitch)
    }
}

pub fn body_aabb(feet: Vec3, height: f32) -> Aabb {
    Aabb::new(
        feet - Vec3::new(HALF_WIDTH, 0.0, HALF_WIDTH),
        feet + Vec3::new(HALF_WIDTH, height, HALF_WIDTH),
    )
}

pub fn view_rotation(yaw: f32, pitch: f32) -> Quat {
    Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0)
}

pub fn look_dir(yaw: f32, pitch: f32) -> Vec3 {
    view_rotation(yaw, pitch) * Vec3::NEG_Z
}

/// Yaw that looks along the horizontal component of `dir`.
pub fn yaw_towards(dir: Vec3) -> f32 {
    (-dir.x).atan2(-dir.z)
}

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    Shot { shooter: FighterId, from: Vec3, to: Vec3 },
    Hit { shooter: FighterId, victim: FighterId, damage: i32 },
    Killed { killer: FighterId, victim: FighterId },
    Respawned { id: FighterId },
}

#[derive(Clone, Debug)]
pub struct World {
    pub tick: u64,
    pub map: Map,
    pub fighters: Vec<Fighter>,
    pub rng: Rng,
    next_id: FighterId,
}

impl World {
    pub fn new(map: Map, seed: u64) -> Self {
        Self { tick: 0, map, fighters: Vec::new(), rng: Rng::new(seed), next_id: 1 }
    }

    pub fn spawn(&mut self, name: impl Into<String>, team: Team) -> FighterId {
        let id = self.next_id;
        self.next_id += 1;
        let pos = self.pick_spawn(team);
        self.fighters.push(Fighter {
            id,
            name: name.into(),
            team,
            pos,
            prev_pos: pos,
            vel: Vec3::ZERO,
            yaw: yaw_towards(-pos),
            pitch: 0.0,
            on_ground: false,
            crouching: false,
            health: MAX_HEALTH,
            respawn_in: 0.0,
            fire_cooldown: 0.0,
            kills: 0,
            deaths: 0,
        });
        id
    }

    pub fn fighter(&self, id: FighterId) -> Option<&Fighter> {
        self.fighters.iter().find(|f| f.id == id)
    }

    /// Advance one tick. Fighters without an input keep their view angles and stand still.
    pub fn step(&mut self, inputs: &[(FighterId, InputCmd)]) -> Vec<Event> {
        let mut events = Vec::new();
        self.tick += 1;

        for i in 0..self.fighters.len() {
            let f = &self.fighters[i];
            let input = inputs
                .iter()
                .find(|(id, _)| *id == f.id)
                .map(|(_, cmd)| *cmd)
                .unwrap_or(InputCmd { yaw: f.yaw, pitch: f.pitch, ..Default::default() });
            self.fighters[i].prev_pos = self.fighters[i].pos;

            if !self.fighters[i].alive() {
                self.tick_dead(i, &mut events);
                continue;
            }
            self.apply_movement(i, &input);
            self.fighters[i].fire_cooldown = (self.fighters[i].fire_cooldown - DT).max(0.0);
            if input.fire && self.fighters[i].fire_cooldown == 0.0 {
                self.fighters[i].fire_cooldown = GUN_INTERVAL;
                self.fire(i, &mut events);
            }
            if self.fighters[i].pos.y < KILL_PLANE {
                let id = self.fighters[i].id;
                self.kill(i, id, &mut events);
            }
        }
        events
    }

    fn tick_dead(&mut self, i: usize, events: &mut Vec<Event>) {
        self.fighters[i].respawn_in -= DT;
        if self.fighters[i].respawn_in > 0.0 {
            return;
        }
        let pos = self.pick_spawn(self.fighters[i].team);
        let f = &mut self.fighters[i];
        f.pos = pos;
        f.prev_pos = pos;
        f.vel = Vec3::ZERO;
        f.yaw = yaw_towards(-pos);
        f.pitch = 0.0;
        f.crouching = false;
        f.health = MAX_HEALTH;
        f.fire_cooldown = 0.0;
        events.push(Event::Respawned { id: f.id });
    }

    fn apply_movement(&mut self, i: usize, input: &InputCmd) {
        let map = &self.map;
        let f = &mut self.fighters[i];
        f.yaw = input.yaw;
        f.pitch = input.pitch.clamp(-1.5, 1.5);

        if input.crouch {
            f.crouching = true;
        } else if f.crouching && !map.collides(&body_aabb(f.pos, STAND_HEIGHT)) {
            f.crouching = false;
        }

        let forward = Vec3::new(-f.yaw.sin(), 0.0, -f.yaw.cos());
        let right = Vec3::new(f.yaw.cos(), 0.0, -f.yaw.sin());
        let wish = (forward * input.movement.y + right * input.movement.x).normalize_or_zero();
        let speed = if f.crouching {
            CROUCH_SPEED
        } else if input.sprint && input.movement.y > 0.0 {
            SPRINT_SPEED
        } else {
            WALK_SPEED
        };
        let target = wish * speed;
        if f.on_ground {
            f.vel.x = target.x;
            f.vel.z = target.z;
            if input.jump {
                f.vel.y = JUMP_SPEED;
            }
        } else {
            let blend = (AIR_CONTROL * DT).min(1.0);
            f.vel.x += (target.x - f.vel.x) * blend;
            f.vel.z += (target.z - f.vel.z) * blend;
        }
        f.vel.y = (f.vel.y - GRAVITY * DT).max(-MAX_FALL_SPEED);

        // Resolve one axis at a time so fighters slide along walls.
        f.on_ground = false;
        let height = f.height();
        for axis in [1, 0, 2] {
            let delta = f.vel[axis] * DT;
            if delta == 0.0 {
                continue;
            }
            f.pos[axis] += delta;
            for block in &map.blocks {
                if !block.aabb.overlaps(&body_aabb(f.pos, height)) {
                    continue;
                }
                const SKIN: f32 = 1e-4;
                let (lo, hi) = match axis {
                    1 => (0.0, height),
                    _ => (HALF_WIDTH, HALF_WIDTH),
                };
                if delta > 0.0 {
                    f.pos[axis] = block.aabb.min[axis] - hi - SKIN;
                } else {
                    f.pos[axis] = block.aabb.max[axis] + lo + SKIN;
                    if axis == 1 {
                        f.on_ground = true;
                    }
                }
                f.vel[axis] = 0.0;
            }
        }
        // Standing exactly on a surface: the downward probe above may not have moved far enough
        // to overlap, so check a thin slab under the feet.
        if !f.on_ground && f.vel.y <= 0.0 {
            let probe = Aabb::new(
                f.pos - Vec3::new(HALF_WIDTH, 0.01, HALF_WIDTH),
                f.pos + Vec3::new(HALF_WIDTH, 0.0, HALF_WIDTH),
            );
            f.on_ground = map.collides(&probe);
        }
    }

    fn fire(&mut self, i: usize, events: &mut Vec<Event>) {
        let shooter = &self.fighters[i];
        let (shooter_id, shooter_team) = (shooter.id, shooter.team);
        let from = shooter.eye();
        let dir = shooter.look_dir();
        let mut t = self.map.raycast(from, dir, GUN_RANGE);
        let mut victim = None;
        for (j, other) in self.fighters.iter().enumerate() {
            if j == i || !other.alive() {
                continue;
            }
            if let Some(hit_t) = other.aabb().ray_hit(from, dir, t) {
                t = hit_t;
                victim = Some(j);
            }
        }
        events.push(Event::Shot { shooter: shooter_id, from, to: from + dir * t });

        // Teammates absorb the shot but take no damage.
        let Some(j) = victim.filter(|&j| self.fighters[j].team != shooter_team) else {
            return;
        };
        let target = &mut self.fighters[j];
        target.health -= GUN_DAMAGE;
        events.push(Event::Hit { shooter: shooter_id, victim: target.id, damage: GUN_DAMAGE });
        if target.health <= 0 {
            self.kill(j, shooter_id, events);
        }
    }

    fn kill(&mut self, i: usize, killer: FighterId, events: &mut Vec<Event>) {
        let victim = &mut self.fighters[i];
        victim.health = 0;
        victim.respawn_in = RESPAWN_SECS;
        victim.deaths += 1;
        victim.vel = Vec3::ZERO;
        let victim_id = victim.id;
        if killer != victim_id
            && let Some(k) = self.fighters.iter_mut().find(|f| f.id == killer)
        {
            k.kills += 1;
        }
        events.push(Event::Killed { killer, victim: victim_id });
    }

    /// A random spawn point at least `SAFE_DIST` from every living enemy, or failing that the
    /// one farthest from them.
    fn pick_spawn(&mut self, team: Team) -> Vec3 {
        const SAFE_DIST: f32 = 14.0;
        let enemies: Vec<Vec3> =
            self.fighters.iter().filter(|f| f.alive() && f.team != team).map(|f| f.pos).collect();
        let clearance = |p: Vec3| enemies.iter().map(|e| e.distance(p)).fold(f32::INFINITY, f32::min);
        let safe: Vec<Vec3> =
            self.map.spawns.iter().copied().filter(|&p| clearance(p) >= SAFE_DIST).collect();
        if !safe.is_empty() {
            return safe[self.rng.below(safe.len())];
        }
        self.map
            .spawns
            .iter()
            .copied()
            .max_by(|a, b| clearance(*a).total_cmp(&clearance(*b)))
            .unwrap_or(Vec3::ZERO)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world_with(team_a: Team, team_b: Team) -> (World, FighterId, FighterId) {
        let mut w = World::new(Map::arena(), 7);
        let a = w.spawn("a", team_a);
        let b = w.spawn("b", team_b);
        (w, a, b)
    }

    fn settle(w: &mut World, ticks: usize) {
        for _ in 0..ticks {
            w.step(&[]);
        }
    }

    #[test]
    fn spawns_are_clear_of_geometry() {
        let map = Map::arena();
        for s in &map.spawns {
            assert!(!map.collides(&body_aabb(*s, STAND_HEIGHT)), "spawn {s} is inside a block");
        }
    }

    #[test]
    fn fighters_land_on_the_floor_and_jump() {
        let (mut w, a, _) = world_with(Team::Humans, Team::Bots);
        settle(&mut w, 30);
        let f = w.fighter(a).unwrap();
        assert!(f.on_ground);
        assert!(f.pos.y.abs() < 0.01, "feet at {}", f.pos.y);

        let jump = InputCmd { yaw: f.yaw, jump: true, ..Default::default() };
        w.step(&[(a, jump)]);
        settle(&mut w, 10);
        assert!(w.fighter(a).unwrap().pos.y > 0.5);
        settle(&mut w, 60);
        assert!(w.fighter(a).unwrap().on_ground);
    }

    #[test]
    fn walls_stop_movement() {
        let (mut w, a, _) = world_with(Team::Humans, Team::Bots);
        // Run at the east wall (inner face at x = 24) for a few seconds.
        let east = InputCmd { yaw: -std::f32::consts::FRAC_PI_2, movement: Vec2::Y, ..Default::default() };
        for _ in 0..600 {
            w.step(&[(a, east)]);
        }
        let f = w.fighter(a).unwrap();
        assert!((f.pos.x - (24.0 - HALF_WIDTH)).abs() < 0.01, "x = {}", f.pos.x);
    }

    #[test]
    fn crouching_lowers_eye_and_cannot_stand_under_a_bar() {
        let (mut w, a, _) = world_with(Team::Humans, Team::Bots);
        w.fighters[0].pos = Vec3::new(0.0, 0.0, 21.5);
        w.fighters[0].crouching = true;
        settle(&mut w, 5);
        assert!(w.fighter(a).unwrap().crouching, "stood up into the bar");

        w.fighters[0].pos = Vec3::new(0.0, 0.0, 10.0);
        settle(&mut w, 1);
        assert!(!w.fighter(a).unwrap().crouching);
    }

    #[test]
    fn shooting_an_enemy_kills_and_respawns_them() {
        let (mut w, a, b) = world_with(Team::Humans, Team::Bots);
        w.fighters[0].pos = Vec3::new(-20.0, 0.0, -10.0);
        w.fighters[1].pos = Vec3::new(-20.0, 0.0, -20.0);
        settle(&mut w, 30);
        let aim = InputCmd { yaw: 0.0, fire: true, ..Default::default() };
        let mut killed = false;
        for _ in 0..120 {
            let events = w.step(&[(a, aim)]);
            killed |= events.contains(&Event::Killed { killer: a, victim: b });
        }
        assert!(killed);
        assert_eq!(w.fighter(a).unwrap().kills, 1);
        settle(&mut w, (RESPAWN_SECS * TICK_HZ as f32) as usize + 1);
        assert_eq!(w.fighter(b).unwrap().health, MAX_HEALTH);
    }

    #[test]
    fn teammates_take_no_damage() {
        let (mut w, a, b) = world_with(Team::Bots, Team::Bots);
        w.fighters[0].pos = Vec3::new(-20.0, 0.0, -10.0);
        w.fighters[1].pos = Vec3::new(-20.0, 0.0, -20.0);
        settle(&mut w, 30);
        let aim = InputCmd { yaw: 0.0, fire: true, ..Default::default() };
        for _ in 0..120 {
            w.step(&[(a, aim)]);
        }
        assert_eq!(w.fighter(b).unwrap().health, MAX_HEALTH);
    }
}
