import yaml

def gen_cfg(cfg_name, array_size=31, camera_saturation=100000000.0, 
detail='Test image in RGB', itertype='spiral', led_gap=5, matsize=32,
max_photon_flux=100000.0, n_iter=5, na=0.17, na_syn=0.95, overlap=15, 
patch=300, pixel_size=2.4e-06, sample_height=130, sample_shape=[800, 800],
shape='point', sim_type='var_pssin', wavelength=4.5e-07, x=2):
    config_dict = {'sim_type': sim_type, 
    'x': x, 
    'na': na, 
    'na_syn': na_syn, 
    'wavelength': wavelength, 
    'pixel_size': pixel_size, 
    'n_iter': n_iter, 
    'shape': shape, 
    'itertype': itertype, 
    'camera_saturation': camera_saturation,  
    'max_photon_flux': max_photon_flux, 
    'sample_height': sample_height,
    'led_gap': led_gap, 'patch': patch, 
    'sample_shape': sample_shape, 
    'array_size': array_size, 
    'matsize': matsize, 
    'overlap': overlap, 
    'detail': detail}
    config_file = open(cfg_name + ".yaml", 'w')
    yaml.dump(config_dict, config_file)
