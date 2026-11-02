//! Bevy wiring: runs `game-core` at a fixed tick, feeds it keyboard/mouse and bot inputs, and
//! renders the result. All game rules live in `game-core`; this crate only reads its state.

// Bevy systems take their dependencies as parameters.
#![allow(clippy::too_many_arguments)]

mod hud;
mod view;

use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use bevy::window::CursorGrabMode;
use bevy::window::{CursorOptions, PrimaryWindow};
use game_core::{BotBrain, Event, FighterId, InputCmd, Map, TICK_HZ, Team, World};

const BOT_COUNT: u64 = 4;
const MOUSE_SENSITIVITY: f32 = 0.0022;

#[derive(Resource)]
pub struct Game {
    pub world: World,
    pub me: FighterId,
    brains: Vec<BotBrain>,
}

impl Game {
    fn new(seed: u64) -> Self {
        let mut world = World::new(Map::arena(), seed);
        let me = world.spawn("you", Team::Humans);
        let brains = (0..BOT_COUNT)
            .map(|i| BotBrain::new(world.spawn(format!("bot{}", i + 1), Team::Bots), seed ^ (i + 1)))
            .collect();
        Self { world, me, brains }
    }

    pub fn name(&self, id: FighterId) -> &str {
        self.world.fighter(id).map_or("?", |f| f.name.as_str())
    }
}

/// The local player's view angles. Updated every frame from the mouse, so looking around is
/// not limited to the tick rate; the sim picks them up on the next tick.
#[derive(Resource, Default)]
pub struct LocalView {
    pub yaw: f32,
    pub pitch: f32,
}

/// Sim events produced in `FixedUpdate`, drained by the view and HUD in `Update`.
#[derive(Resource, Default)]
pub struct PendingEvents(pub Vec<Event>);

fn main() {
    let seed = 0x5EED;
    let game = Game::new(seed);
    let me = game.world.fighter(game.me).unwrap();
    let view = LocalView { yaw: me.yaw, pitch: me.pitch };

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "web-fps".into(),
                canvas: Some("#bevy".into()),
                fit_canvas_to_parent: true,
                prevent_default_event_handling: true,
                ..default()
            }),
            ..default()
        }))
        .insert_resource(Time::<Fixed>::from_hz(TICK_HZ))
        .insert_resource(game)
        .insert_resource(view)
        .init_resource::<PendingEvents>()
        .add_plugins((view::ViewPlugin, hud::HudPlugin))
        .add_systems(Update, (grab_cursor, mouse_look).chain())
        .add_systems(FixedUpdate, step_sim)
        .add_systems(Last, |mut pending: ResMut<PendingEvents>| pending.0.clear())
        .run();
}

/// Whether the mouse is captured for looking around.
pub fn pointer_locked(cursor: &CursorOptions) -> bool {
    // The page owns pointer lock on the web, so ask it.
    #[cfg(target_arch = "wasm32")]
    {
        let _ = cursor;
        web_sys::window()
            .and_then(|w| w.document())
            .is_some_and(|d| d.pointer_lock_element().is_some())
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        cursor.grab_mode == CursorGrabMode::Locked
    }
}

/// Native only: capture the mouse on click, release on Escape. On the web the page requests
/// pointer lock itself (see index.html), because browsers only grant it inside the click handler,
/// and the browser releases it on Escape.
#[cfg(not(target_arch = "wasm32"))]
fn grab_cursor(
    mut cursor: Single<&mut CursorOptions, With<PrimaryWindow>>,
    mouse: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
) {
    if mouse.just_pressed(MouseButton::Left) {
        cursor.grab_mode = CursorGrabMode::Locked;
        cursor.visible = false;
    }
    if keys.just_pressed(KeyCode::Escape) {
        cursor.grab_mode = CursorGrabMode::None;
        cursor.visible = true;
    }
}

#[cfg(target_arch = "wasm32")]
fn grab_cursor() {}

fn mouse_look(
    motion: Res<AccumulatedMouseMotion>,
    cursor: Single<&CursorOptions, With<PrimaryWindow>>,
    mut view: ResMut<LocalView>,
) {
    if !pointer_locked(&cursor) {
        return;
    }
    view.yaw -= motion.delta.x * MOUSE_SENSITIVITY;
    view.pitch = (view.pitch - motion.delta.y * MOUSE_SENSITIVITY).clamp(-1.5, 1.5);
}

fn step_sim(
    mut game: ResMut<Game>,
    mut view: ResMut<LocalView>,
    mut pending: ResMut<PendingEvents>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    cursor: Single<&CursorOptions, With<PrimaryWindow>>,
) {
    // Solo play pauses while the mouse is released (menu up, tab in background).
    let active = pointer_locked(&cursor);
    if !active {
        return;
    }
    let game = &mut *game;
    let mut inputs: Vec<(FighterId, InputCmd)> =
        game.brains.iter_mut().map(|b| (b.id, b.think(&game.world))).collect();

    let axis = |pos: KeyCode, neg: KeyCode| keys.pressed(pos) as i32 as f32 - keys.pressed(neg) as i32 as f32;
    inputs.push((
        game.me,
        InputCmd {
            movement: Vec2::new(axis(KeyCode::KeyD, KeyCode::KeyA), axis(KeyCode::KeyW, KeyCode::KeyS)),
            yaw: view.yaw,
            pitch: view.pitch,
            jump: keys.pressed(KeyCode::Space),
            // Not Ctrl: Ctrl+W would close the browser tab.
            crouch: keys.pressed(KeyCode::KeyC),
            sprint: keys.pressed(KeyCode::ShiftLeft),
            fire: mouse.pressed(MouseButton::Left),
        },
    ));

    let events = game.world.step(&inputs);
    // On respawn the sim picks a new facing; adopt it.
    if events.contains(&Event::Respawned { id: game.me }) {
        let me = game.world.fighter(game.me).unwrap();
        (view.yaw, view.pitch) = (me.yaw, me.pitch);
    }
    pending.0.extend(events);
}
