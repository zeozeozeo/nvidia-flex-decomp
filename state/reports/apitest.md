# apitest matrix

* shipped (golden): `C:\Code\nvidia-flex-decomp\work\apitest\shipped`
* recon: `C:\Code\nvidia-flex-decomp\work\apitest\recon`

| scenario | shipped exit | recon exit | verdict |
|---|---|---|---|
| init_lifecycle | 0 | 0 | FAIL: init_desc |
| solver_lifecycle | 0 | 0 | match |
| params_roundtrip | 0 | 0 | match |
| buffers_basic | 0 | 0 | match |
| particles_setget | 0 | 0 | match |
| shapes_meshes | 0 | 0 | match |
| rigids_springs_inflatables | 0 | 0 | match |
| dynamic_triangles_contacts | 0 | 0 | FAIL: get_dynamic_triangles_normals, get_contacts_counts |
| bounds_update | 0 | 0 | match |
| timers_debug | 0 | 0 | match |
| context_flush | 0 | 0 | match |
| register_probes | 0 | 0 | match |
| device_dll | 0 | 0 | match |
| sim_fluid | 0 | 0 | FAIL: f00_particles_hash, f00_velocities_hash, f00_particles, f00_velocities, f00_smooth, f00_anisotropy_q1, f00_anisotropy_q2, f00_anisotropy_q3, f01_particles_hash, f01_velocities_hash, f02_particles_hash, f02_velocities_hash, f03_particles_hash, f03_velocities_hash, f04_particles_hash, f04_velocities_hash, f04_particles, f04_velocities, f04_smooth, f04_anisotropy_q1, f04_anisotropy_q2, f04_anisotropy_q3, f05_particles_hash, f05_velocities_hash, f06_particles_hash, f06_velocities_hash, f07_particles_hash, f07_velocities_hash |
| sim_fluid_simple | 0 | 0 | match |
| sim_solid | 0 | 0 | FAIL: f00_particles_hash, f00_velocities_hash, f00_particles, f00_velocities, f00_smooth, f01_particles_hash, f01_velocities_hash, f01_particles, f01_velocities, f01_smooth, f02_particles_hash, f02_velocities_hash, f02_particles, f02_velocities, f02_smooth, f03_particles_hash, f03_velocities_hash, f03_particles, f03_velocities, f03_smooth, f04_particles_hash, f04_velocities_hash, f04_particles, f04_velocities, f04_smooth, f05_particles_hash, f05_velocities_hash, f05_particles, f05_velocities, f05_smooth |
| sim_diffuse | 0 | 0 | FAIL: f00_particles_hash, f00_velocities_hash, f00_particles, f00_velocities, f00_smooth, f00_anisotropy_q1, f01_particles_hash, f01_velocities_hash, f01_particles, f01_velocities, f01_smooth, f01_anisotropy_q1, f02_particles_hash, f02_velocities_hash, f02_particles, f02_velocities, f02_smooth, f02_anisotropy_q1, f03_particles_hash, f03_velocities_hash, f03_particles, f03_velocities, f03_smooth, f03_anisotropy_q1, f04_particles_hash, f04_velocities_hash, f04_particles, f04_velocities, f04_smooth, f04_anisotropy_q1, f05_particles_hash, f05_velocities_hash, f05_particles, f05_velocities, f05_smooth, f05_anisotropy_q1 |
| sim_dynamic_triangles | 0 | 0 | FAIL: f00_particles_hash, f00_velocities_hash, f00_particles, f00_velocities, f00_normals, f00_anisotropy_q2, f01_particles_hash, f01_velocities_hash, f01_particles, f01_velocities, f01_normals, f01_anisotropy_q2, f02_particles_hash, f02_velocities_hash, f02_particles, f02_velocities, f02_normals, f02_anisotropy_q2, f03_particles_hash, f03_velocities_hash, f03_particles, f03_velocities, f03_normals, f03_anisotropy_q2, f04_particles_hash, f04_velocities_hash, f04_particles, f04_velocities, f04_normals, f04_anisotropy_q2, get_dynamic_triangles_normals |
| ext_cloth | 0 | 0 | FAIL: f00_particles, f01_particles, f02_particles, f03_particles, f04_particles |
| ext_rigid | 0 | 0 | FAIL: f00_particles, f00_shape_positions, f00_shape_rotations, f01_particles, f01_shape_positions, f01_shape_rotations, f02_particles, f02_shape_positions, f02_shape_rotations, f03_particles, f03_shape_positions, f03_shape_rotations, f04_particles, f04_shape_positions, f04_shape_rotations |
| ext_soft | 0 | 0 | FAIL: skinning_weights, f00_particles, f01_particles, f02_particles, f03_particles, f04_particles, f05_particles_after_update_instances, f05_shape_positions, f05_shape_rotations |
| ext_tearing | 0 | 3221226505 | match |
| ext_forcefield | 0 | 0 | FAIL: f00_particles, f00_particles_hash, f01_particles, f01_particles_hash, f02_particles, f02_particles_hash, f03_particles, f03_particles_hash, f04_particles, f04_particles_hash |
| ext_movingframe | 0 | 0 | FAIL: f00_particles, f01_particles, f02_particles, f03_particles, f04_particles |
| ext_container_full | 0 | 0 | FAIL: f00_particles_inst1, f00_particles_inst2, f01_particles_inst1, f01_particles_inst2, f02_particles_inst1, f02_particles_inst2 |

**178 failing records.**
