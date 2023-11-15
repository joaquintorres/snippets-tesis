import shutil
import base64
import numpy as np
from io import BytesIO
from flask import Flask, render_template, request, Response, redirect

import gigapixel.acquire
import gigapixel.interfaces
import gigapixel
import pathlib
import json
import time
import threading

from gigapixel import config
PATH = pathlib.Path(__file__).parent

DUMMY = False
DRY_RUN = False
BIT_DEPTH = config.BIT_DEPTH
PORT = config.PORT
ROI = config.ROI

camera_ready = threading.Event()
acquisition_stopped = threading.Event()

camera_ready.set()
acquisition_stopped.set()

camera = gigapixel.set_camera(
    dummy=DUMMY,
    bit_depth=BIT_DEPTH,
    roi=ROI
)

illumination = gigapixel.set_illumination(dummy=DUMMY, port=PORT)

setup = gigapixel.FpmAcquisitionSetup.make_default()
setup.dummy_devices = DUMMY
setup.dry_run = DRY_RUN

status = gigapixel.FpmStatus(setup)

app = Flask(
    "gigapixel",
    template_folder=str(PATH / "web"),
    static_url_path="",
    static_folder=str(PATH / "web")
)


@app.route("/")
def html_home():
    return redirect("/status")


@app.route("/status", methods=["GET", "POST"])
def html_status():
    if request.method == "GET":
        return render_template(
            "status.html",
            enabled=acquisition_stopped.is_set(),
            user="Operator",
            active="status",
        )
    elif request.method == "POST":
        response = json.dumps({
            "progress": status.progress,
            "eta": status.eta
        })
        print(status.progress, status.eta)
        return Response(response, status=200, mimetype='application/json')


@app.route("/acquire")
def html_acquire():
    if not acquisition_stopped.is_set():
        return redirect("/status")

    sample_state = "success"
    if setup.description == "" and setup.name == "":
        sample_state = "warning"
    if setup.name != "":
        if (setup.sample_dir / setup.name).exists():
            sample_state = "danger"

    eta = setup.get_estimated_time()
    free_space = shutil.disk_usage(setup.sample_dir).free / 1e9
    required_space = setup.get_estimated_size()

    illumination_state = "success"
    if required_space > free_space:
        illumination_state = "danger"

    n_leds = setup.get_scheme_sequence().shape[0]

    return render_template(
        "acquire.html",
        enabled=acquisition_stopped.is_set(),
        user="Operator",
        active="acquire",
        sample_state=sample_state,
        camera_state="success",
        illumination_state=illumination_state,
        name=setup.name,
        description=setup.description,
        exp_scheme="Gaussian",
        scheme=setup.scheme.capitalize(),
        colors=" - ".join([c.upper() for c in setup.colors]),
        red_exp=f"{setup.red_exp}",
        green_exp=f"{setup.green_exp}",
        blue_exp=f"{setup.blue_exp}",
        n_leds=f"{n_leds} (in a {setup.size}x{setup.size} area)",
        eta_hours=int(np.floor(eta / 3600)),
        eta_minutes=int(np.floor((eta % 3600) / 60)),
        free_space_gb=free_space,
        request_space_gb=required_space
    )


@app.route("/camera", methods=['POST', 'GET'])
def html_camera():
    if request.method == 'GET':
        if not acquisition_stopped.is_set():
            return redirect("/status")
        return render_template(
            "camera.html",
            enabled=acquisition_stopped.is_set(),
            active="camera",
            user="Operator",
            exp_min=50,
            exp_max=1e5,
            exp_r=setup.red_exp,
            exp_g=setup.green_exp,
            exp_b=setup.blue_exp,
        )
    elif request.method == 'POST':
        if not acquisition_stopped.is_set():
            return Response(status=500)
        data = json.loads(request.data)
        setup.red_exp = int(data["r_exp"])
        setup.green_exp = int(data["g_exp"])
        setup.blue_exp = int(data["b_exp"])
        return Response(status=200)


@app.route("/focus")
def html_focus():
    if not acquisition_stopped.is_set():
        return redirect("/status")
    return render_template(
        "focus.html",
        enabled=acquisition_stopped.is_set(),
        active="focus",
        user="Operator",
        exp=(setup.red_exp + setup.green_exp + setup.blue_exp) // 3,
        x=setup.fpm_config.center[1],
        y=setup.fpm_config.center[0],
    )

@app.route("/sample", methods=['POST', 'GET'])
def html_sample():
    if request.method == 'GET':
        if not acquisition_stopped.is_set():
            return redirect("/status")
        return render_template(
            "sample.html",
            enabled=acquisition_stopped.is_set(),
            active="sample",
            user="Operator",
            name=setup.name,
            description=setup.description,
        )
    elif request.method == 'POST':
        if not acquisition_stopped.is_set():
            return Response(status=500)
        data = json.loads(request.data)
        setup.name = data["name"]
        setup.description = data["description"]
        return Response(status=200)


@app.route("/illumination", methods=['POST', 'GET'])
def html_illumination():
    if request.method == 'GET':
        if not acquisition_stopped.is_set():
            return redirect("/status")
        return render_template(
            "illumination.html",
            enabled=acquisition_stopped.is_set(),
            active="illumination",
            user="Operator",
            size=setup.size,
            scheme=setup.scheme,
            red="r" in setup.colors,
            green="g" in setup.colors,
            blue="b" in setup.colors,
        )
    elif request.method == 'POST':
        if not acquisition_stopped.is_set():
            return Response(status=500)
        data = json.loads(request.data)
        setup.size = int(data["size"])
        setup.scheme = data["scheme"]
        colors = [data["red"], data["green"], data["blue"]]
        setup.colors = "".join([
            color
            for color, flag in zip("rgb", colors)
            if flag is True
        ])
        eta = setup.get_estimated_time()
        free_space = shutil.disk_usage(setup.sample_dir).free / 1e9
        response = json.dumps({
            "eta_hours": np.floor(eta / 3600),
            "eta_minutes": np.floor((eta % 3600) / 60),
            "free_space_gb": f"{free_space:.2f}",
            "request_space_gb": f"{setup.get_estimated_size():.2f}"
        })
        return Response(response, status=200, mimetype='application/json')


@app.route("/histogram/<color>/<exposure>")
def get_histogram(color, exposure):
    if not acquisition_stopped.is_set():
        return redirect("/status")
    camera_ready.wait()
    camera_ready.clear()
    camera.set_gain_exposure(gain=100, exposure=int(exposure))
    illumination.turn_on_led(
        setup.fpm_config.center[0],
        setup.fpm_config.center[1],
        color=color
    )
    frame = camera.get_frame()
    illumination.turn_off_leds()
    camera_ready.set()

    fig = gigapixel.fig_histogram(frame, color, f"Exposure {exposure} us")

    buf = BytesIO()
    fig.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img style='width: 90%;' src='data:image/png;base64,{data}'/>"


@app.route("/pattern/<size>/<scheme>/<colors>")
def get_illumination(size, scheme, colors):
    if not acquisition_stopped.is_set():
        return redirect("/status")
    fig = gigapixel.fig_scheme(int(size), gigapixel.get_scheme(scheme))
    buf = BytesIO()
    fig.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img style='width: 90%;' src='data:image/png;base64,{data}'/>"


@app.route("/capture/<color>/<exposure>/<led_x>/<led_y>")
def get_image(color, exposure, led_x, led_y):
    if not acquisition_stopped.is_set():
        return redirect("/status")
    camera_ready.wait()
    camera_ready.set()
    camera.set_gain_exposure(gain=100, exposure=int(setup.red_exp))
    illumination.turn_on_led(int(led_y), int(led_x), color=color)
    frame = camera.get_frame()
    illumination.turn_off_leds()
    camera_ready.set()

    fig = gigapixel.fig_camera(frame, color)

    buf = BytesIO()
    fig.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return (
        f"<img class='text-center' style='width: 100%; height: 100%' "
        f"src='data:image/png;base64,{data}'/>"
    )


@app.route("/start", methods=["POST"])
def start():
    if not acquisition_stopped.is_set():
        return redirect("/status")

    camera_ready.wait()

    required_space = setup.get_estimated_size()
    free_space = shutil.disk_usage(setup.sample_dir).free / 1e9

    if required_space > free_space:
        return Response(status=500, response="Insufficient space in disk")

    if setup.name == "":
        if setup.description == "":
            return Response(status=500, response="Empty description")
        else:
            setup.name = time.strftime("%Y%m%d-%H%M%S")

    (setup.sample_dir / setup.name).mkdir(parents=True, exist_ok=False)

    status.recalculate()
    sequence_iterator = gigapixel.acquire_sequence(
        camera=camera,
        illumination=illumination,
        setup=setup,
        status=status
    )
    status.running = True
    thread = threading.Thread(target=acquire, args=[sequence_iterator])
    thread.start()
    acquisition_stopped.clear()
    return Response(status=200)


def acquire(sequence_iterator):
    for image in sequence_iterator:
        if setup.dry_run:
            print(image.get_name())
        else:
            current_name = image.save(setup.sample_dir / setup.name)
            print(current_name)
    acquisition_stopped.set()
