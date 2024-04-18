use crossbeam_channel::unbounded;
use fastrand::Rng;
use rocketsim_rs::{
    autocxx::WithinUniquePtr,
    cxx::UniquePtr,
    glam_ext::glam::{Mat3A, Vec3A},
    math::{Angle, Vec3},
    sim::{Arena, ArenaMemWeightMode, CarConfig, CarControls, GameMode, Team},
};
use std::{
    f32::consts::PI,
    fs,
    io::{self, Write},
    thread,
    time::{Duration, Instant},
};
use zstd::stream::copy_encode;

const INTERVAL_TIME: Duration = Duration::from_secs(300);
const OUT_FOLDER: &str = "results";

fn main() {
    rocketsim_rs::init(None);

    fs::create_dir_all(OUT_FOLDER).unwrap();

    let (tx, rx) = unbounded();

    let num_threads = thread::available_parallelism().unwrap().into();

    for _ in 0..num_threads {
        let tx = tx.clone();
        thread::spawn(move || {
            let mut simulation = Simulation::new();
            let mut initial_allocation_num = 4096;

            loop {
                let mut results = Vec::with_capacity(initial_allocation_num);
                let interval_start_time = Instant::now();

                while interval_start_time.elapsed() < INTERVAL_TIME {
                    if let Some(result) = simulation.do_random() {
                        results.push(result);
                    }
                }

                initial_allocation_num = results.capacity();
                tx.send(results).unwrap();
            }
        });
    }

    drop(tx);

    let start_time = Instant::now();
    let mut total_time = 0.;

    let mut num_iters = fs::read_dir(OUT_FOLDER).unwrap().count();
    println!("Starting with the name {num_iters}.bin for the next file");

    let mut current_threads = 0;
    let mut current_results = Vec::new();

    for results in rx {
        current_threads += 1;
        total_time += results.iter().map(|r| r.time).sum::<f32>();
        current_results.extend(results);

        if current_threads == num_threads {
            current_threads = 0;

            // print a quick performance update
            let hours_gathered = total_time / 3600.;
            let hours_per_second = hours_gathered / start_time.elapsed().as_secs_f32();
            print!(
                "Total time simulated: {:.2} days ({hours_per_second:.1} hps)\r",
                hours_gathered / 24.
            );
            io::stdout().flush().unwrap();

            // write current_results to file

            // f32 = 4 bytes, 7 f32 per result
            let size = current_results.len() * 4 * 7;
            let mut bytes = Vec::with_capacity(size);

            for result in &current_results {
                let iav = result.initial_angular_velocity;
                bytes.extend(iav.x.to_le_bytes());
                bytes.extend(iav.y.to_le_bytes());
                bytes.extend(iav.z.to_le_bytes());

                let rt = result.relative_target;
                bytes.extend(rt.pitch.to_le_bytes());
                bytes.extend(rt.yaw.to_le_bytes());
                bytes.extend(rt.roll.to_le_bytes());

                bytes.extend(result.time.to_le_bytes());
            }

            let file_name = format!("{OUT_FOLDER}/{}.bin", num_iters);
            num_iters += 1;

            // create the file and file writer
            let file = fs::File::create(&file_name).unwrap();
            let mut writer = io::BufWriter::new(file);

            // compress the data
            copy_encode(&bytes[..], &mut writer, 3).unwrap();
            // write the compressed data
            writer.flush().unwrap();

            current_results.clear();
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct SimResult {
    initial_angular_velocity: Vec3A,
    relative_target: Angle,
    time: f32,
}

struct Simulation {
    arena: UniquePtr<Arena>,
    car_id: u32,
    rng: Rng,
}

impl Simulation {
    fn new() -> Self {
        let mut arena = Arena::new(GameMode::THE_VOID, ArenaMemWeightMode::HEAVY, 120.).within_unique_ptr();

        let mut mutators = arena.get_mutator_config();
        mutators.gravity.z = -f32::EPSILON;

        arena.pin_mut().set_mutator_config(mutators);

        Self {
            car_id: arena.pin_mut().add_car(Team::BLUE, CarConfig::octane()),
            rng: Rng::new(),
            arena,
        }
    }

    fn do_random(&mut self) -> Option<SimResult> {
        let mut ball_state = self.arena.pin_mut().get_ball();
        ball_state.pos.z = -1000.;
        self.arena.pin_mut().as_mut().set_ball(ball_state);

        let mut car_state = self.arena.pin_mut().as_mut().get_car(self.car_id);

        car_state.pos = Vec3::ZERO;
        car_state.vel = Vec3::ZERO;

        // random initial angular velocity
        let mut ang_vel = Vec3A::new(self.rng.f32(), self.rng.f32(), self.rng.f32());
        ang_vel = ang_vel.normalize() * self.rng.f32() * 5.5;
        car_state.ang_vel = ang_vel.into();

        // random initial orientation
        let angle = Angle {
            pitch: self.rng.f32() * PI,
            yaw: self.rng.f32() * PI,
            roll: self.rng.f32() * PI,
        };

        car_state.rot_mat = angle.to_rotmat();
        let initial_rot = Mat3A::from(car_state.rot_mat);

        let relative_ang_vel = initial_rot.transpose() * ang_vel;

        self.arena.pin_mut().set_car(self.car_id, car_state).unwrap();

        let target_pitch = self.rng.f32() * PI;
        let target_yaw = self.rng.f32() * PI;

        // target angles relative to car angle
        let rel_target_angles = Angle {
            pitch: target_pitch - angle.pitch,
            yaw: target_yaw - angle.yaw,
            roll: 0. - angle.roll,
        };

        // angles to target
        // x = forward, y = right, z = up
        let target = Vec3A::new(
            target_pitch.cos() * target_yaw.cos(),
            target_pitch.sin(),
            target_pitch.cos() * target_yaw.sin(),
        ) * 1000.;

        let target_dir = target.normalize();

        let mut num_steps = 0;
        loop {
            let car_state = self.arena.pin_mut().get_car(self.car_id);

            // check if the angle is < 0.1 rad
            let rot = Mat3A::from(car_state.rot_mat);
            let forward = rot * Vec3A::X;
            let angle = forward.dot(target_dir).clamp(-1., 1.).acos();

            if angle < 0.1 {
                break;
            }

            if num_steps > 120 * 30 {
                // this doesn't happen but just in case
                println!("Failed to reach target?");
                return None;
            }

            let local_target = rot.transpose() * target;
            let local_ang_vel = rot.transpose() * Vec3A::from(car_state.ang_vel);
            let local_up = rot * Vec3A::Z;

            let controls = default_pd(local_target, local_ang_vel, local_up);
            self.arena.pin_mut().set_car_controls(self.car_id, controls).unwrap();

            self.arena.pin_mut().step(1);
            num_steps += 1;
        }

        let time = num_steps as f32 / 120.;
        Some(SimResult {
            initial_angular_velocity: relative_ang_vel,
            relative_target: rel_target_angles,
            time,
        })
    }
}

fn control_pd(angle: f32, rate: f32) -> f32 {
    ((35. * (angle + rate)).powi(3) / 10.).clamp(-1., 1.)
}

fn default_pd(local_target: Vec3A, local_ang_vel: Vec3A, local_up: Vec3A) -> CarControls {
    let target_angles = Angle {
        pitch: local_target.z.atan2(local_target.x),
        yaw: local_target.y.atan2(local_target.x),
        roll: local_up.y.atan2(local_up.z),
    };

    let pitch = control_pd(target_angles.pitch, local_ang_vel.y / 3.4);
    let yaw = control_pd(target_angles.yaw, -local_ang_vel.z / 5.0);
    let roll = control_pd(target_angles.roll, local_ang_vel.x / 3.1);

    CarControls {
        pitch,
        yaw,
        roll,
        ..Default::default()
    }
}
