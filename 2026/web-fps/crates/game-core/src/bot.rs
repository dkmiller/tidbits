//! Bot AI. A bot is only an input producer: it reads the world and returns an [`InputCmd`],
//! exactly like a human at a keyboard would.
//!
//! v0 behavior: run straight at the nearest enemy, jump and sidestep when stuck on geometry,
//! and once in line of sight close to a preferred range, strafe and fire with aim error.

use std::f32::consts::{PI, TAU};

use bevy_math::{Vec2, Vec3};

use crate::rng::Rng;
use crate::sim::{DT, FighterId, InputCmd, World, yaw_towards};

/// Max turn speed, radians per second.
const TURN_RATE: f32 = 5.0;
/// Seconds of continuous line of sight before the first shot.
const REACTION_SECS: f32 = 0.45;
/// Max random aim offset, radians, re-rolled a few times per second.
const AIM_ERROR: f32 = 0.06;
/// Seconds between trigger pulls.
const TRIGGER_GAP: (f32, f32) = (0.35, 0.9);
const MAX_ENGAGE_DIST: f32 = 45.0;

#[derive(Clone, Debug)]
pub struct BotBrain {
    pub id: FighterId,
    rng: Rng,
    preferred_range: f32,
    reaction: f32,
    trigger_cooldown: f32,
    aim_offset: Vec2,
    aim_timer: f32,
    strafe: f32,
    strafe_timer: f32,
    stuck_timer: f32,
    unstick_timer: f32,
    unstick_dir: f32,
}

impl BotBrain {
    pub fn new(id: FighterId, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let preferred_range = rng.range(6.0, 14.0);
        Self {
            id,
            rng,
            preferred_range,
            reaction: 0.0,
            trigger_cooldown: 0.0,
            aim_offset: Vec2::ZERO,
            aim_timer: 0.0,
            strafe: 0.0,
            strafe_timer: 0.0,
            stuck_timer: 0.0,
            unstick_timer: 0.0,
            unstick_dir: 1.0,
        }
    }

    pub fn think(&mut self, world: &World) -> InputCmd {
        let Some(me) = world.fighter(self.id) else {
            return InputCmd::default();
        };
        let idle = InputCmd { yaw: me.yaw, pitch: me.pitch, ..Default::default() };
        if !me.alive() {
            self.reaction = 0.0;
            return idle;
        }
        let Some(target) = world
            .fighters
            .iter()
            .filter(|f| f.alive() && f.team != me.team)
            .min_by(|a, b| a.pos.distance_squared(me.pos).total_cmp(&b.pos.distance_squared(me.pos)))
        else {
            self.reaction = 0.0;
            return idle;
        };

        let (eye, target_eye) = (me.eye(), target.eye());
        let to = target_eye - eye;
        let dist = to.length();
        let los = world.map.line_of_sight(eye, target_eye);

        // Aim: turn toward the target at a limited rate, with a wandering offset.
        self.aim_timer -= DT;
        if self.aim_timer <= 0.0 {
            self.aim_timer = self.rng.range(0.25, 0.6);
            self.aim_offset = Vec2::new(
                self.rng.range(-AIM_ERROR, AIM_ERROR),
                self.rng.range(-AIM_ERROR, AIM_ERROR) * 0.5,
            );
        }
        let desired_yaw = yaw_towards(to) + self.aim_offset.x;
        let desired_pitch = to.y.atan2(Vec2::new(to.x, to.z).length()) + self.aim_offset.y;
        let yaw_err = wrap_angle(desired_yaw - me.yaw);
        let max_turn = TURN_RATE * DT;
        let yaw = me.yaw + yaw_err.clamp(-max_turn, max_turn);
        let pitch = me.pitch + (desired_pitch - me.pitch).clamp(-max_turn, max_turn);

        // Move: chase until in sight and in range, then strafe.
        let chase = !los || dist > self.preferred_range;
        let mut movement = Vec2::ZERO;
        let mut jump = false;
        if chase {
            movement.y = 1.0;
            let horizontal_speed = Vec3::new(me.vel.x, 0.0, me.vel.z).length();
            if me.on_ground && horizontal_speed < 1.0 && yaw_err.abs() < 0.5 {
                self.stuck_timer += DT;
            } else {
                self.stuck_timer = 0.0;
            }
            if self.stuck_timer > 0.2 {
                self.stuck_timer = 0.0;
                jump = true;
                self.unstick_timer = self.rng.range(0.5, 1.0);
                self.unstick_dir = if self.rng.unit() < 0.5 { -1.0 } else { 1.0 };
            }
        } else {
            self.strafe_timer -= DT;
            if self.strafe_timer <= 0.0 {
                self.strafe_timer = self.rng.range(0.6, 1.6);
                self.strafe = [-1.0, 0.0, 1.0][self.rng.below(3)];
            }
            movement.x = self.strafe;
        }
        if self.unstick_timer > 0.0 {
            self.unstick_timer -= DT;
            movement.x = self.unstick_dir;
        }

        // Fire: after a reaction delay, in short taps, only when roughly on target.
        if los && dist < MAX_ENGAGE_DIST {
            self.reaction += DT;
        } else {
            self.reaction = 0.0;
        }
        self.trigger_cooldown -= DT;
        let fire = self.reaction >= REACTION_SECS && self.trigger_cooldown <= 0.0 && yaw_err.abs() < 0.15;
        if fire {
            self.trigger_cooldown = self.rng.range(TRIGGER_GAP.0, TRIGGER_GAP.1);
        }

        InputCmd { movement, yaw, pitch, jump, crouch: false, sprint: chase && dist > 15.0, fire }
    }
}

fn wrap_angle(a: f32) -> f32 {
    (a + PI).rem_euclid(TAU) - PI
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::Map;
    use crate::sim::Team;

    /// A human that stands still in the open should eventually be found and killed by bots.
    #[test]
    fn bots_hunt_down_a_stationary_human() {
        let mut world = World::new(Map::arena(), 42);
        let human = world.spawn("you", Team::Humans);
        let mut brains: Vec<BotBrain> = (0..3)
            .map(|i| {
                let id = world.spawn(format!("bot{i}"), Team::Bots);
                BotBrain::new(id, 100 + i)
            })
            .collect();

        let mut human_died = false;
        for _ in 0..(60 * 60) {
            let inputs: Vec<_> = brains.iter_mut().map(|b| (b.id, b.think(&world))).collect();
            human_died |= world.step(&inputs).iter().any(|e| {
                matches!(e, crate::sim::Event::Killed { victim, .. } if *victim == human)
            });
            if human_died {
                break;
            }
        }
        assert!(human_died, "bots never killed the human in 60 seconds");
    }

    #[test]
    fn wrap_angle_stays_in_range() {
        assert!((wrap_angle(3.0 * PI) - PI).abs() < 1e-4 || (wrap_angle(3.0 * PI) + PI).abs() < 1e-4);
        assert!((wrap_angle(0.5) - 0.5).abs() < 1e-6);
        assert!((wrap_angle(-0.5 - TAU) + 0.5).abs() < 1e-4);
    }
}
