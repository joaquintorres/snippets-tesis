import numpy as np
from chanoscope.chanoscope.drivers.camera import CameraQHY

CAMERA_ID = 0
RGB_MATRIX_PORT = "/dev/ttyACM0"

qhy = CameraQHY(bit_depth=14)
qhy.open_resource(CAMERA_ID)
exposure_range = np.arange(200, 1000, 50)  # TODO
gain = 100

def exposure_sweep(exp_range, gain, qhy_camera):
    raw_images = []
    for exposure in exp_range:
        qhy_camera.set_gain_exposure(gain, exposure)
        img = qhy_camera.single_get_frame()
        raw_images.append(img)
    return raw_images

for color in ["r", "g", "b"]:
    driver.turn_on_led(central_led, color)
    sweep_images = np.array(
        exposure_sweep(exp_range=exposure_range, gain=gain, qhy_camera=qhy)
    )
    np.save("exposure_sweep_gain_{}_{}".format(gain, color), sweep_images)
