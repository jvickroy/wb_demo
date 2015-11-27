
# Wise Ball Device Simulation #
### - Software Design and Implementation ###

    date   : 11/23/2015 5:51:19 PM 
    author : j. vickroy
    e-mail : jgv.home@gmail.com


### Statement of Work ###

Contractor will provide development and consulting services necessary to create a wireless transfer device prototype.

The wireless transfer device will utilize a camera to detect its motion relative to a target device (PC or laptop) in order to smoothly control a cursor displayed on the target device.

The prototype will be able to wirelessly copy/transfer a user selected file between two target devices.  The wireless transfer device will provide onscreen cursor control to the second target device after successful transfer of the selected file.

The prototype will provide the functionality described above between the two target machines interchangeably.

Contractor will report and electronically submit weekly progress to the Company’s private repository on Friday of each week. Contractor will be responsible for providing a brief description of specific weekly objectives, on or before Monday of the given work week.

### Implementation ###

The Wise Ball Simulator and Target device interface are written in Python (v2.7) and comprise the following files:

- `WiseBall.py` : a simulation of the capabilities of a Wise Ball hardware device
- `Target.py` : a device to be controlled by a Wise Ball
- `utils.py` : generic utility components

Startup configuration is provided by the **`WiseBall.ini`** file.

`WiseBall.py`, `utils.py`, and `WiseBall.ini` must be accessible to the machine hosting the Wise Ball simulator.  `Target.py`, `utils.py`, and `WiseBall.ini` must be accessible to each Target machine.

The above files need not be installed in the Python distribution folder; placing them in a single folder is sufficient.

Additionally, the above files rely on external packages described in the **Dependencies** section.


### Dependencies ###

The **`WiseBall.py`** host machine requires installation of the following packages in the listed order:  

1. [Python 2.7](http://www.python.org)
1. [Numpy](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy) for numerical array processing
2. [Scipy](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy) for mathematical algorithms
1. [OpenCV (cv2)](http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv) for video processing
2. [Enthought Traits](http://www.lfd.uci.edu/~gohlke/pythonlibs/#ets) for type definition and validation
1. [pywin32](http://sourceforge.net/projects/pywin32/files/pywin32/) for Microsoft Windows specific keyboard interaction
 
Each **`Target.py`** host machine requires installation of the following packages in the listed order:  

1. [Python 2.7](http://www.python.org)
1. [Numpy](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy) for numerical array processing
1. [PyAutoGUI](https://pyautogui.readthedocs.org/en/latest/) for cross-platform, mouse control
2. [Enthought Traits](http://www.lfd.uci.edu/~gohlke/pythonlibs/#ets) for type definition and validation
1. [pywin32](http://sourceforge.net/projects/pywin32/files/pywin32/) for Microsoft Windows specific keyboard interaction

Note, [pywin32](http://sourceforge.net/projects/pywin32/files/pywin32/) is only available for Microsoft Target machines.

### Use ###

`WiseBall.py` and `Target.py` (**Implementation** section) are the top-level scripts to be run from a console window.  **`WiseBall.py`** is to be run only on the Microsoft machine hosting the simulator.  **`Target.py`** is to be run on each Target machine.

Each script is configurable by specifying command-line parameters that override default behaviors and|or by editing entries in **`WiseBall.ini`**.

Information on command-line options is available by running a script with the **`--help`** or **`-h`** option as follows:

        $ python WiseBall.py --help
        $ python Target.py --help

When the **`--help`** option is specified, the scripts present help information and then immediately terminate.

On Microsoft Windows, script execution may be terminated by pressing a keyboard key (default is the **`[Esc]`** key).  Use the **`--stopkey`** command-line option to specify an alternate key.  There is no keyboard support for deployment on a non-Microsoft OS. 

The `WiseBall.py` script uses the **`[UP]`** and **`[DOWN]`** cursor keys to simulate a Wise Ball device button.  These keys behave like push-buttons; once **`[DOWN]`** is pressed, it remains pressed until **`[UP]`** is pressed.  This behavior facilitates mouse drag-n-drop operations.  The initial button state is **`[UP]`**.

When run in non-help mode, each script logs processing events in the console window.  Logging verbosity is controlled by the **`--verbosity`** command-line option.  Supported verbosity levels are; (**`high`**, **`medium`**, **`low`**); the default is **`medium`**.


### User Interface Notes ###

At startup, `WiseBall.py` presents the following real-time displays:

- *Target Identification*
- *Target Monitor Detection*

which show the status of target device acquisition and target monitor location within each video frame.

##### *Target Identification* Display #####

Color is used to identify Targets; each Target must present a unique color (see `[Target <:>]` sections in `WiseBall.ini`).  A threshold percentage of video frame pixels, must match a Target color signature for successful identification. 

The  ***similarity*** slider control alters the threshold percentage to partially compensate for various camera / target monitor distances.  The continuously-updated message, at the bottom of the display, provides guidance on how to adjust the slider.

##### *Target Monitor Detection* Display #####

Edge detection is used to identify target monitor candidates (i.e., quadrilaterals with interior angles *near* 90 degrees) in each video frame.

Five slider controls (*min* and *max area*, *angle*, *min* and *max edge*) provide some degree of real-time control over the edge detection procedure as follows:

- *min* and *max area* may be used to set limits on the size of target monitor candidates to compensate for various camera / target monitor distances.
- *angle* may be used to set allowable deviations from 90 degrees for interior angles of target monitor candidates to compensate for lens / perspective distortions.
- *min* and *max edge* may be used to set limits on constitutes an edge (intensity gradient) in video frame features.

In practice, *min* and *max area* have proven to be the most useful to exclude unwanted quadrilaterals from being identified as the target monitor.

### Notes ###

USB3 / DFK 31AU03 connections are troublesome.  Sometimes at startup, video frames can not be captured; when this happens, `WiseBall.py` script execution must be stopped and the Imaging Source ***IC Capture*** tool must be used to toggle the frame rate setting (e.g., between 15 and 3.75 fps).  Attempts to perform this frame rate toggling, within `WiseBall.py` had no effect.  Experience shows that a 3.75 fps rate is most reliable with USB3 connections.

The DFK 31AU03 / Pentax 4.8mm lens combination delivers fuzzy images.  Images are much sharper with a 28mm Olympus lens.  Consequently, target monitor detection is more reliable with the 28mm lens.

