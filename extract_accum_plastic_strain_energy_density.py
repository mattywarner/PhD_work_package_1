import damask
import numpy as np

def extract_slip_systems(material_file):
    mat = damask.ConfigMaterial.load(material_file)
    crystal_structure = damask.Crystal(lattice = mat['phase']['Cu']['lattice'])

    slip_dirs = crystal_structure.kinematics('slip')['direction'][0]
    slip_norms = crystal_structure.kinematics('slip')['plane'][0]

    return slip_dirs, slip_norms

def update_slip_systems(damask_view, slip_dirs, slip_norms, rve_shape):

    rot_matrices = damask.Rotation(damask_view.get('O')).reshape(rve_shape, order = 'F').as_matrix()

    all_new_slip_dirs = []
    all_new_slip_norms = []

    for i in range(len(rot_matrices)):
        new_slip_dirs = []
        new_slip_norms = []
        for j in range(len(slip_dirs)):
            new_slip_dir = rot_matrices[i] @ slip_dirs[j]
            new_slip_norm = rot_matrices[i] @ slip_norms[j]
            new_slip_dirs.append(new_slip_dir)
            new_slip_norms.append(new_slip_norm)
        
        all_new_slip_dirs.append(new_slip_dirs)
        all_new_slip_norms.append(new_slip_norms)

    return all_new_slip_dirs, all_new_slip_norms

def calc_accum_plastic_strain_energy_density(damask_file, mat_file, rve_shape):
    result = damask.Result(damask_file)

    slip_dirs, slip_norms = extract_slip_systems(mat_file)

    total_plastic_work_arrs = []

    for i in range(len(result.increments)):
        x = result.view(increments = result.increments[i])
        gamma = x.get('gamma_sl').reshape(rve_shape.append(12), order = 'F')

        if i == 0:
            new_slip_dirs, new_slip_norms = slip_dirs, slip_norms

        else:
            new_slip_dirs, new_slip_norms = update_slip_systems(x, slip_dirs, slip_norms, rve_shape)

        sigma = x.get('sigma').reshape(rve_shape.extend([3,3]), order = 'F')

        # Calculate resolved shear stresses
        tau = np.einsum('vij,vsi,vsj->vs', sigma, new_slip_dirs, new_slip_norms)

        plastic_work = np.abs(np.multiply(tau, gamma))

        total_plastic_work_per_inc = np.sum(plastic_work, axis=3)

        if i == 0:
            total_plastic_work_arrs.append(total_plastic_work_per_inc)
        else:
            total_plastic_work_current_time_inc = total_plastic_work_per_inc - total_plastic_work_arrs[i-1]
            total_plastic_work_arrs.append(total_plastic_work_current_time_inc)

    total_plastic_work_arrs = np.array(total_plastic_work_arrs) # Should be shape [12,128,128,128]  

    accum_plastic_work = np.sum(total_plastic_work_arrs, axis = 0)

    return accum_plastic_work



