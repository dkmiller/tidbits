//! 3D view: map blocks, bot bodies, the first-person camera and gun, and shot tracers.

use std::collections::HashMap;

use bevy::prelude::*;
use game_core::sim::{self, STAND_HEIGHT};
use game_core::{Event, FighterId, Material as Surface};

use crate::{Game, LocalView, PendingEvents};

const TRACER_SECS: f32 = 0.07;
const HIT_FLASH_SECS: f32 = 0.12;
/// Gun position in camera space.
const GUN_OFFSET: Vec3 = Vec3::new(0.17, -0.15, -0.55);
const MUZZLE_OFFSET: Vec3 = Vec3::new(0.17, -0.13, -0.76);

pub struct ViewPlugin;

impl Plugin for ViewPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ClearColor(Color::srgb(0.56, 0.71, 0.86)))
            .insert_resource(GlobalAmbientLight { brightness: 600.0, ..default() })
            .init_resource::<Effects>()
            .add_systems(Startup, setup)
            .add_systems(Update, (collect_effects, sync_fighters, update_camera, draw_effects).chain());
    }
}

#[derive(Component)]
struct PlayerCamera {
    eye_height: f32,
}

#[derive(Component)]
struct Gun;

#[derive(Component)]
struct FighterVisual {
    id: FighterId,
    material: Handle<StandardMaterial>,
    flashing: bool,
}

struct Tracer {
    from: Vec3,
    to: Vec3,
    ttl: f32,
    color: Color,
}

#[derive(Resource, Default)]
struct Effects {
    tracers: Vec<Tracer>,
    recoil: f32,
    hit_flash: HashMap<FighterId, f32>,
}

fn surface_color(surface: Surface) -> Color {
    match surface {
        Surface::Floor => Color::srgb(0.32, 0.36, 0.32),
        Surface::Wall => Color::srgb(0.78, 0.78, 0.74),
        Surface::Cover => Color::srgb(0.45, 0.52, 0.62),
        Surface::Crate => Color::srgb(0.6, 0.45, 0.28),
        Surface::Accent => Color::srgb(0.9, 0.55, 0.2),
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    game: Res<Game>,
    view: Res<LocalView>,
) {
    let mut palette = HashMap::new();
    for block in &game.world.map.blocks {
        let material = palette
            .entry(block.material)
            .or_insert_with(|| {
                materials.add(StandardMaterial {
                    base_color: surface_color(block.material),
                    perceptual_roughness: 0.9,
                    ..default()
                })
            })
            .clone();
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::from_size(block.aabb.size()))),
            MeshMaterial3d(material),
            Transform::from_translation(block.aabb.center()),
        ));
    }

    commands.spawn((
        DirectionalLight { illuminance: 9000.0, shadow_maps_enabled: true, ..default() },
        Transform::from_xyz(20.0, 40.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Bots: a capsule body with a dark visor on the -Z (forward) side.
    let body = meshes.add(Capsule3d::new(sim::HALF_WIDTH, STAND_HEIGHT - 2.0 * sim::HALF_WIDTH));
    let visor = meshes.add(Cuboid::new(0.4, 0.12, 0.12));
    let visor_material = materials.add(Color::srgb(0.08, 0.08, 0.1));
    for f in game.world.fighters.iter().filter(|f| f.id != game.me) {
        let material = materials.add(StandardMaterial { base_color: Color::srgb(0.85, 0.2, 0.2), ..default() });
        commands.spawn((
            FighterVisual { id: f.id, material: material.clone(), flashing: false },
            Mesh3d(body.clone()),
            MeshMaterial3d(material),
            Transform::default(),
            children![(
                Mesh3d(visor.clone()),
                MeshMaterial3d(visor_material.clone()),
                Transform::from_xyz(0.0, sim::STAND_EYE - STAND_HEIGHT / 2.0, -0.26),
            )],
        ));
    }

    // First-person camera with the gun attached.
    let gun_material = materials.add(StandardMaterial { base_color: Color::srgb(0.15, 0.15, 0.17), ..default() });
    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection { fov: 80f32.to_radians(), near: 0.05, ..default() }),
        Transform::from_rotation(sim::view_rotation(view.yaw, view.pitch)),
        PlayerCamera { eye_height: sim::STAND_EYE },
        children![(
            Gun,
            Mesh3d(meshes.add(Cuboid::new(0.05, 0.07, 0.4))),
            MeshMaterial3d(gun_material),
            Transform::from_translation(GUN_OFFSET),
        )],
    ));
}

fn collect_effects(
    pending: Res<PendingEvents>,
    game: Res<Game>,
    camera: Single<&Transform, With<PlayerCamera>>,
    mut effects: ResMut<Effects>,
) {
    for event in &pending.0 {
        match *event {
            Event::Shot { shooter, from, to } => {
                let (from, color) = if shooter == game.me {
                    effects.recoil = 1.0;
                    (camera.transform_point(MUZZLE_OFFSET), Color::srgb(1.0, 0.95, 0.5))
                } else {
                    (from, Color::srgb(1.0, 0.45, 0.25))
                };
                effects.tracers.push(Tracer { from, to, ttl: TRACER_SECS, color });
            }
            Event::Hit { victim, .. } => {
                effects.hit_flash.insert(victim, HIT_FLASH_SECS);
            }
            _ => {}
        }
    }
}

fn sync_fighters(
    game: Res<Game>,
    fixed: Res<Time<Fixed>>,
    effects: Res<Effects>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut visuals: Query<(&mut FighterVisual, &mut Transform, &mut Visibility)>,
) {
    let alpha = fixed.overstep_fraction();
    for (mut visual, mut transform, mut visibility) in &mut visuals {
        let Some(f) = game.world.fighter(visual.id) else { continue };
        *visibility = if f.alive() { Visibility::Inherited } else { Visibility::Hidden };
        let height = f.height();
        let feet = f.prev_pos.lerp(f.pos, alpha);
        transform.translation = feet + Vec3::Y * height / 2.0;
        transform.scale = Vec3::new(1.0, height / STAND_HEIGHT, 1.0);
        transform.rotation = Quat::from_rotation_y(f.yaw);

        let flashing = effects.hit_flash.get(&visual.id).is_some_and(|t| *t > 0.0);
        if flashing != visual.flashing {
            visual.flashing = flashing;
            if let Some(mut m) = materials.get_mut(&visual.material) {
                m.emissive = if flashing { LinearRgba::rgb(4.0, 4.0, 4.0) } else { LinearRgba::BLACK };
            }
        }
    }
}

fn update_camera(
    game: Res<Game>,
    view: Res<LocalView>,
    fixed: Res<Time<Fixed>>,
    time: Res<Time>,
    effects: Res<Effects>,
    camera: Single<(&mut Transform, &mut PlayerCamera), Without<Gun>>,
    gun: Single<(&mut Transform, &mut Visibility), With<Gun>>,
) {
    let Some(me) = game.world.fighter(game.me) else { return };
    let (mut transform, mut camera) = camera.into_inner();
    // Ease the eye height so crouching (and dying) is a quick slide, not a snap.
    let target = if me.alive() { me.eye().y - me.pos.y } else { 0.3 };
    let ease = 1.0 - (-18.0 * time.delta_secs()).exp();
    camera.eye_height += (target - camera.eye_height) * ease;
    transform.translation = me.prev_pos.lerp(me.pos, fixed.overstep_fraction()) + Vec3::Y * camera.eye_height;
    transform.rotation = sim::view_rotation(view.yaw, view.pitch);

    let (mut gun_transform, mut gun_visibility) = gun.into_inner();
    *gun_visibility = if me.alive() { Visibility::Inherited } else { Visibility::Hidden };
    gun_transform.translation = GUN_OFFSET + Vec3::new(0.0, 0.01, 0.06) * effects.recoil;
    gun_transform.rotation = Quat::from_rotation_x(0.12 * effects.recoil);
}

fn draw_effects(time: Res<Time>, mut effects: ResMut<Effects>, mut gizmos: Gizmos) {
    let dt = time.delta_secs();
    // Floor grid, so movement reads clearly on the flat floor.
    let grid = Color::srgba(0.0, 0.0, 0.0, 0.18);
    for i in -12..=12 {
        let c = i as f32 * 2.0;
        gizmos.line(Vec3::new(c, 0.01, -24.0), Vec3::new(c, 0.01, 24.0), grid);
        gizmos.line(Vec3::new(-24.0, 0.01, c), Vec3::new(24.0, 0.01, c), grid);
    }
    for t in &mut effects.tracers {
        gizmos.line(t.from, t.to, t.color);
        t.ttl -= dt;
    }
    effects.tracers.retain(|t| t.ttl > 0.0);
    effects.recoil = (effects.recoil - dt * 10.0).max(0.0);
    effects.hit_flash.retain(|_, t| {
        *t -= dt;
        *t > -1.0
    });
}
