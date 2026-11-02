//! Pure FPS simulation: no rendering, windowing, clocks, or I/O.
//!
//! The app feeds one [`InputCmd`] per fighter into [`World::step`] at a fixed tick rate and
//! renders the resulting state. Bots are just another source of `InputCmd`s.

pub mod bot;
pub mod map;
pub mod rng;
pub mod sim;

pub use bot::BotBrain;
pub use map::{Aabb, Map, Material};
pub use sim::{DT, Event, Fighter, FighterId, InputCmd, TICK_HZ, Team, World};
