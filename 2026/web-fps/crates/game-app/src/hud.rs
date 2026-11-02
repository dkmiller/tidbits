//! Screen-space UI: crosshair, health, score, kill feed, damage flash, and prompts.

use bevy::prelude::*;
use bevy::window::{CursorOptions, PrimaryWindow};
use game_core::Event;

use crate::{Game, PendingEvents, pointer_locked};

const HIT_MARKER_SECS: f32 = 0.15;
const DAMAGE_FLASH_SECS: f32 = 0.35;
const FEED_SECS: f32 = 5.0;
const HELP: &str = "WASD move   Shift sprint   Space jump   C crouch\nClick shoot   Tab scores   Esc release mouse";

pub struct HudPlugin;

impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<HudState>()
            .add_systems(Startup, setup)
            .add_systems(Update, update_hud);
    }
}

#[derive(Component, Clone, Copy, PartialEq, Eq)]
enum HudText {
    Health,
    Score,
    Feed,
    Center,
}

#[derive(Component)]
struct Crosshair;

#[derive(Component)]
struct DamageVignette;

#[derive(Resource, Default)]
struct HudState {
    hit_marker: f32,
    damage: f32,
    feed: Vec<(String, f32)>,
}

fn text(kind: HudText, size: f32) -> impl Bundle {
    (kind, Text::default(), TextFont::from_font_size(size), TextColor(Color::WHITE), TextShadow::default())
}

fn setup(mut commands: Commands) {
    let full_screen = Node {
        position_type: PositionType::Absolute,
        width: Val::Percent(100.0),
        height: Val::Percent(100.0),
        ..default()
    };
    commands.spawn((DamageVignette, full_screen.clone(), BackgroundColor(Color::NONE)));
    commands.spawn((
        Node { justify_content: JustifyContent::Center, align_items: AlignItems::Center, ..full_screen.clone() },
        children![(Crosshair, Text::new("+"), TextFont::from_font_size(30.0), TextColor(Color::WHITE))],
    ));
    commands.spawn((
        Node {
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Center,
            padding: UiRect::top(Val::Percent(28.0)),
            ..full_screen
        },
        children![(text(HudText::Center, 26.0), TextLayout { justify: Justify::Center, ..default() })],
    ));
    let corner = |left: bool, top: bool| Node {
        position_type: PositionType::Absolute,
        left: if left { Val::Px(16.0) } else { Val::Auto },
        right: if left { Val::Auto } else { Val::Px(16.0) },
        top: if top { Val::Px(12.0) } else { Val::Auto },
        bottom: if top { Val::Auto } else { Val::Px(12.0) },
        ..default()
    };
    commands.spawn((corner(true, false), children![text(HudText::Health, 32.0)]));
    commands.spawn((corner(false, true), children![text(HudText::Score, 22.0)]));
    commands.spawn((corner(true, true), children![text(HudText::Feed, 18.0)]));
}

fn update_hud(
    time: Res<Time>,
    game: Res<Game>,
    pending: Res<PendingEvents>,
    keys: Res<ButtonInput<KeyCode>>,
    cursor: Single<&CursorOptions, With<PrimaryWindow>>,
    mut state: ResMut<HudState>,
    mut texts: Query<(&HudText, &mut Text)>,
    mut crosshair: Single<&mut TextColor, With<Crosshair>>,
    mut vignette: Single<&mut BackgroundColor, With<DamageVignette>>,
) {
    let dt = time.delta_secs();
    for event in &pending.0 {
        match *event {
            Event::Hit { shooter, .. } if shooter == game.me => state.hit_marker = HIT_MARKER_SECS,
            Event::Hit { victim, .. } if victim == game.me => state.damage = DAMAGE_FLASH_SECS,
            Event::Killed { killer, victim } => {
                let line = if killer == victim {
                    format!("{} fell out of the world", game.name(victim))
                } else {
                    format!("{}  >  {}", game.name(killer), game.name(victim))
                };
                state.feed.push((line, FEED_SECS));
            }
            _ => {}
        }
    }
    state.hit_marker = (state.hit_marker - dt).max(0.0);
    state.damage = (state.damage - dt).max(0.0);
    state.feed.iter_mut().for_each(|(_, t)| *t -= dt);
    state.feed.retain(|(_, t)| *t > 0.0);
    let excess = state.feed.len().saturating_sub(5);
    state.feed.drain(..excess);

    crosshair.0 = if state.hit_marker > 0.0 { Color::srgb(1.0, 0.25, 0.2) } else { Color::WHITE };
    vignette.0 = Color::srgba(0.8, 0.0, 0.0, 0.35 * state.damage / DAMAGE_FLASH_SECS);

    let Some(me) = game.world.fighter(game.me) else { return };
    let center = if !pointer_locked(&cursor) {
        format!("Click to play\n\n{HELP}")
    } else if !me.alive() {
        format!("You died. Respawning in {:.1}s", me.respawn_in.max(0.0))
    } else if keys.pressed(KeyCode::Tab) {
        let mut rows: Vec<_> = game.world.fighters.iter().collect();
        rows.sort_by(|a, b| b.kills.cmp(&a.kills).then(a.deaths.cmp(&b.deaths)));
        rows.iter().map(|f| format!("{:<6} {:>3} K {:>3} D", f.name, f.kills, f.deaths)).collect::<Vec<_>>().join("\n")
    } else {
        String::new()
    };

    for (kind, mut text) in &mut texts {
        let value = match kind {
            HudText::Health => format!("+ {}", me.health.max(0)),
            HudText::Score => format!("Kills {}   Deaths {}", me.kills, me.deaths),
            HudText::Feed => state.feed.iter().map(|(l, _)| l.as_str()).collect::<Vec<_>>().join("\n"),
            HudText::Center => center.clone(),
        };
        if text.0 != value {
            text.0 = value;
        }
    }
}
