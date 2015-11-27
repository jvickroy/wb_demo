'''
CONTENTS
    components implementing a Wise Ball pointer simulator

NOTES
    * This module relies on a Microsoft-specific keyboard interface.
    * This module may be imported (e.g., for unit testing) or run as a script from a command line.
    * If run as a command-line script (see __main__ at the bottom), it simulates the behaviors of a
        Wise Ball pointer.

AUTHOR
    jgv.home@gmail.com
'''

# standard Python packages ...
import logging, math, os, socket
from   time import sleep
# generic extensions to standard Python ...
import cv2   # http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv
import numpy # http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
from   traits.api import Any, Array, Bool, Constant, Dict, Enum, File, Float, Function, HasTraits, Instance, Int, List, Property, Range, String, Trait, Tuple # http://www.lfd.uci.edu/~gohlke/pythonlibs/#ets
from   traits.api import cached_property, undefined          # http://www.lfd.uci.edu/~gohlke/pythonlibs/#ets
from   win32con   import VK_UP, VK_DOWN, VK_ESCAPE, VK_SPACE # [MS Windows Specific] UP/DOWN arrow and [Esc] keys -- VK is virtual key :: http://sourceforge.net/projects/pywin32/files/pywin32/
# application-specific Python modules ...
from   utils import MOUSE_DIRECTIVE, SHOW_SIGNATURE, WITHDRAW_SIGNATURE, Bounds, Button, Color, Directive, Keyboard, Percentage, Position, Quadrilateral, Ratio, multi_method, pairs_of


class NoCamera  (Exception) : pass
class NoImage   (Exception) : pass
class NoMonitor (Exception) : pass
class NoTarget  (Exception) : pass


class Target (HasTraits):

    class Monitor (HasTraits):
        width  = Range (low= 100, high=5000, value=undefined, exclude_low=False, exclude_high=False, desc='monitor width (pixels)')
        height = Range (low= 100, high=5000, value=undefined, exclude_low=False, exclude_high=False, desc='monitor height (pixels)')
        def __str__ (self):
            return '{}(width:{}, height:{})'.format (self.__class__.__name__, self.width, self.height)

    class Signature (HasTraits):
        color     = String (desc='color signature name')
        dimmest   = Color  (desc='dimmest allowable RGB intensities')
        brightest = Color  (desc='brightest allowable RGB intensities')
        def __eq__ (self, other):
            return self.dimmest == other.dimmest or self.brightest == other.brightest
        def __ne__ (self, other):
            return self.dimmest != other.dimmest and self.brightest != other.brightest
        def __hash__ (self):
            return hash ((hash(self.dimmest), hash(self.brightest))) # color signature is all that matters for identification purposes
        def __setattr__ (self, attribute, setting):
            if attribute == 'dimmest' and hasattr (self,'brightest'):
                assert self.brightest != setting, '{} `dimmest` and `brightest` attributes are equal ({})'.format(self.__class__.__name__,setting)
            elif attribute == 'brightest' and hasattr (self,'dimmest'):
                assert self.dimmest != setting,   '{} `dimmest` and `brightest` attributes are equal ({})'.format(self.__class__.__name__,setting)
            HasTraits.__setattr__(self, attribute, setting)
        def __str__ (self):
            return '{}(color:{}, dimmest:{}, brightest:{})'.format (self.__class__.__name__, self.color, self.dimmest, self.brightest)

    # external specifications ...
    name      = String   (desc='target machine name on local network for transmission of directives')
    signature = Instance (Signature, desc='Target color signature for identification by Wise Ball')
    monitor   = Instance (Monitor  , desc='Target screen')
    port      = Range    (low=3000, high=5000, value=undefined, exclude_low=False, exclude_high=False, desc='socket communications port number')
    # internal specifications ...
    log    = Instance (logging.Logger) # must be created externally if each Target is to have a uniquely-named logger
    #log    = Instance (logging.Logger, factory=logging.getLogger, args=(name,))       `name` is a traits.trait_types.String object here
    #log    = Instance (logging.Logger, factory=logging.getLogger, args=(name.value,)) `name.value`is `None` here

    @classmethod
    def create (_self_, target, settings):
        name       = target.split(' <:> ')[1] # [Section] name suffix (after <:>) assumed to be the machine name on local network
        width      = settings.getint (target, 'width')
        height     = settings.getint (target, 'height')
        port       = settings.getint (target, 'port')
        color      = settings.get    (target, 'color')
        dimmest    = settings.get    (target, 'dimmest')
        brightest  = settings.get    (target, 'brightest')
        dimmest    = Color (**dict (zip (('Red','Green','Blue'), [int(this) for this in dimmest  .split(',')])))
        brightest  = Color (**dict (zip (('Red','Green','Blue'), [int(this) for this in brightest.split(',')])))
        assert brightest != dimmest, '{} `dimmest` and `brightest` attributes are equal ({})'.format(self.__name__,dimmest)
        signature  = _self_.Signature (color=color, brightest=brightest, dimmest=dimmest)
        monitor    = _self_.Monitor   (width=width, height=height)
        log        = logging.getLogger (name)
        parameters = dict (zip (('name','color','signature','monitor','port','log'),(name,color,signature,monitor,port,log)))
        return _self_ (**parameters)

    def __eq__ (self, other):
        return self.name == other.name or self.signature == other.signature or self.port == other.port

    def __ne__ (self, other):
        return self.name != other.name and self.signature != other.signature and self.port != other.port

    def __hash__ (self):
        # required procedure if Target instances are to be `set` elements -- `set`s are useful to prevent duplicate elements
        return hash (self.signature) # color signature is all that matters for identification purposes

    def __str__ (self):
        return '{}({})'.format(self.__class__.__name__, self.name)


class Image (HasTraits):

    # externally-managed parameters ...
    source = Trait (0, Range(0,9), File, desc='image source (camera|file)')
    data   = Array (dtype=numpy.uint8, desc='image pixel intensity values')
    # internally-managed parameters ...
    width   = Property (depends_on='data', desc='image width (pixels)')
    height  = Property (depends_on='data', desc='image height (pixels)')
    isColor = Property (depends_on='data', desc='boolean flag (True only if data is RGB color)')
    shape   = Property (depends_on='data', desc='image dimensions (pixels)')
    size    = Property (depends_on='data', desc='total number of image pixels')
    log     = Instance (logging.Logger, factory=logging.getLogger, args=('Image',))

    @cached_property
    def _get_width (self):
        return self.data.shape[1]

    @cached_property
    def _get_height (self):
        return self.data.shape[0]

    @cached_property
    def _get_isColor (self):
        return self.data.ndim == 3

    @cached_property
    def _get_shape (self):
        return self.data.size

    @cached_property
    def _get_size (self):
        return self.width * self.height

    def __str__ (self):
        result = 'video device #{}'.format(self.source) if isinstance (self.source, int) else self.source
        return result


class Camera (HasTraits):

    # externally-managed parameters ...
    source  = Trait (0, Range(0,9), File, desc='source of video frames -- connected hardware device or file')
    rate    = Enum  (4, (4,8,15), desc='capture rate (frames/second)')
    retries = Dict  (key_trait=String, value_trait=Int, desc='retry upper limits for camera functions')
    # internally-managed parameters ...
    #device  = Function   (desc='cv2 device|file handle') # interpreter, runtime inspection says it is: built-in function VideoCapture but using a Function declaration raises:
                # traits.trait_errors.TraitError: The 'device' trait of a Camera instance must be a function, but a value of <VideoCapture 0000000003ACCDF0> <type 'cv2.VideoCapture'> was specified.
    #device  = Instance   (cv2.VideoCapture, desc='cv2 device|file handle') # http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html says it is: cv2.VideoCapture instance but using an Instance delcaration raises:
                # traits.trait_errors.TraitError: The 'device' trait of a Camera instance must be a builtin_function_or_method or None, but a value of <VideoCapture 00000000039FAE10> <type 'cv2.VideoCapture'> was specified.
    device  = Any      (desc='cv2 device|file handle')
    width   = Property (depends_on='device'       , desc='camera sensor width (pixels)')
    height  = Property (depends_on='device'       , desc='camera sensor height (pixels)')
    size    = Property (depends_on='width, height', desc='total number of camera sensor pixels')
    center  = Property (depends_on='width, height', desc='center point (pixels) of camera sensor')
    frame   = Instance (Image                     , desc='current image')
    log     = Instance (logging.Logger, factory=logging.getLogger, args=('Camera',))

    @classmethod
    def create (_self_, settings):
        section  = 'Camera'
        source   = settings.get    (section,'source')
        rate     = settings.getint (section,'rate')
        connects = settings.getint (section,'connects')
        captures = settings.getint (section,'captures')
        try:
            source = int (source)
        except ValueError:
            pass
        retries    = dict (zip (('connect','capture'),(connects,captures)))
        parameters = dict (zip (('source','rate','retries'),(source,rate,retries)))
        return _self_ (**parameters)

    def __str__ (self):
        return '{}(source:{}, width:{}, height:{}, FPS:{})'.format (self.__class__.__name__, self.source, self.width, self.height, self.rate) if self.device else '{}(disconnected)'.format (self.__class__.__name__)

    def connect (self):
        if self.connected() : self.disconnect()
        attempts = self.retries['connect'] + 1
        for attempt in range (attempts):
            self.device = cv2.VideoCapture (self.source)
            if self.connected() : break
            sleep (1) # seconds
        else:
            raise NoCamera ('video camera source ({}) not reachable in {} attempts'.format (self.source, attempts))
        self.device.set (cv2.cv.CV_CAP_PROP_FPS, self.rate)
        self.log.info (' %s connected', str(self))

    def connected (self):
        try:
            result = self.device.isOpened()
        except AttributeError:
            result = False
        return result

    disconnect = multi_method.dispatcher()

    @multi_method.procedure (disconnect, lambda *args, **kwargs : args[0].device)
    def disconnect (self):
        try:
            self.device.release()
        except:
            pass
        assert not self.connected(), 'disconnect failure -- still connected'
        del self.device
        self.log.info (' disconnected')

    @multi_method.procedure (disconnect, lambda *args, **kwargs : not args[0].device)
    def disconnect (self):
        # self.log.debug (' disconnect request ignored :: device not connected')
        pass

    def capture (self):
        self.frame = None
        assert self.connected(), 'video camera is disconnected'
        attempts = self.retries['capture'] + 1
        for attempt in range (attempts):
            success, frame = self.device.read() # `frame` typically is numpy.ndarray but may be `None`
            if success : break
        else:
            raise NoImage ('Camera source ({}) frame-grab not successful in {} attempts -- jogging frame rate, in IC Capture, typically fixes this'.format (self.source, attempts)) # jogging frame rate in this procedure did not work
        self.frame = Image (source=self.source, data=frame)

    def _set_rate (self, value):
        value = int (round(value))
        self.device.set (cv2.cv.CV_CAP_PROP_FPS, value)
        self.rate = int (round (self.device.get (cv2.cv.CV_CAP_PROP_FPS)))

    def _validate_rate (self, value):
        assert int(value) in (3,7,15) or float(value) in (3.75, 7.5, 15.0)

    @cached_property
    def _get_rate (self):
        return int (round (self.device.get (cv2.cv.CV_CAP_PROP_FPS))) if self.device else None

    @cached_property
    def _get_width (self):
        return int (self.device.get (cv2.cv.CV_CAP_PROP_FRAME_WIDTH)) if self.device else None

    @cached_property
    def _get_height (self):
        return int (self.device.get (cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)) if self.device else None

    @cached_property
    def _get_size (self):
        return self.width * self.height if self.device else None

    @cached_property
    def _get_center (self):
        return Position (x=self.width/2, y=self.height/2) if self.device else None


class Detector (HasTraits):
    """ quadrilaterals (e.g., Target monitor outline) detector in an image (e.g., video frame) """

    # externally-managed parameters ...
    Filters     = Enum (('angle','area','gradient'))
    constraints = Dict (key_trait=Filters, value_trait=(Percentage,Bounds,Range(low=0, high=30)), desc='constraits on what constitutes a detected Target monitor')
    # internally-managed parameters ...
    image          = Instance (Image        , desc='image being analyzed for presence of Target monitor')
    quadrilaterals = Dict     (key_trait=Filters, value_trait=List(numpy.ndarray), desc='vertices of `image` quadrilaterals subject to constraints')
    rads_per_deg   = Constant ((2*math.pi) / 360.0, desc='number of radians in 1 degree of angle')
    areas          = List (Range(low=0.0, high=100.0), desc='quadrilaterals areas as ascending percent of image size')
    log            = Instance (logging.Logger, factory=logging.getLogger, args=('Monitor Detector',))

    @classmethod
    def create (_self_, settings):
        section      = 'Target Monitor Detection'
        angle_min    = settings.getint   (section,'min angle')
        area_min     = settings.getfloat (section,'min area')
        area_max     = settings.getfloat (section,'max area')
        gradient_min = settings.getint (section,'min gradient')
        gradient_max = settings.getint (section,'max gradient')
        assert area_min     < area_max    , '[{}] section initialization error :: "min area" ({}) not less than "max area" ({})'        .format (section, area_min, area_max)
        assert gradient_min < gradient_max, '[{}] section initialization error :: "min gradient" ({}) not less than "max gradient" ({})'.format (section, gradient_min, gradient_max)
        area         = Percentage (minimum=area_min, maximum=area_max)
        gradient     = Bounds (minimum=gradient_min, maximum=gradient_max)
        constraints  = dict (angle=angle_min, area=area, gradient=gradient)
        return _self_ (constraints=constraints)

    def process (self, image):
        """
        PURPOSE
            determine the vertices of the largest quadrilateral in `image` that satisfies brightness gradient and quadrilateral area constraints
        PARAMETERS
            * image : Image instance to be analyzed
        RETURNS
            nothing
        """
        self.image = image
        self.quadrilaterals.update (dict(angle=list(), area=list(), gradient=list()))
        self.apply (self.image                     , 'gradient')
        self.apply (self.quadrilaterals['gradient'], 'area'    )
        self.apply (self.quadrilaterals['area'    ], 'angle'   )

    apply = multi_method.dispatcher()

    @multi_method.procedure (apply, lambda *args, **kwargs : args[2] == 'angle')
    def apply (self, quadrilaterals, constraint):
        # retain only quadrilaterals whose interior angles are "close" to 90 degrees ...
        self.quadrilaterals[constraint] = list()
        nVertices = 4 # number of vertices in a quadrilateral
        threshold = math.cos (self.as_radians (90-self.constraints[constraint]))
        for quadrilateral in quadrilaterals:
            vertices = quadrilateral.reshape (-1, 2) # `quadrilateral` has shape (4,1,2) :: `vertices` has shape (4,2)
            max_cos   = numpy.max ([self._angle_cos_(vertices[i], vertices[(i+1) % nVertices], vertices[(i+2) % nVertices]) for i in xrange(nVertices)])
            if -threshold <= max_cos <= threshold : self.quadrilaterals[constraint].append (quadrilateral)
        if not self.quadrilaterals[constraint]: raise NoMonitor ('image contains no features satisfying target monitor interior {} constraint of {} degrees'.format (constraint, self.constraints[constraint]))
        self.log.debug ('image contains %d quadrilaterals satisfying target monitor interior %s constraint of %d degrees', len(self.quadrilaterals[constraint]), constraint, self.constraints[constraint])
        
    @multi_method.procedure (apply, lambda *args, **kwargs : args[2] == 'area')
    def apply (self, quadrilaterals, constraint):
        self.quadrilaterals[constraint] = list()
        del self.areas
        areas    = [cv2.contourArea (quadrilateral) for quadrilateral in quadrilaterals] # pixels^2
        areas    = [100*(area/self.image.size) for area in areas]                        # percentage of image size
        self.log.debug ('smallest and largest quadrilaterals occupy %.1f%% and %.1f%% of image', min(areas), max(areas)) if areas else False
        min_area = self.constraints[constraint].minimum
        max_area = self.constraints[constraint].maximum
        features = [(quadrilateral,area) for (quadrilateral,area) in zip (quadrilaterals, areas) if min_area <= area <= max_area] # filtered by area constraint
        features = [(quadrilateral,area) for (quadrilateral,area) in sorted (features, key = lambda (quadrilateral,area): area) ] # ordered by increasing area
        self.quadrilaterals[constraint] = [quadrilateral for (quadrilateral,area) in features] # ordered by increasing percentage of image size
        self.areas                      = [area          for (quadrilateral,area) in features] # ascending order
        if not self.quadrilaterals[constraint]: raise NoMonitor ('image contains no features satisfying target monitor {} constraint {}'.format (constraint, self.constraints[constraint]))
        self.log.debug ('image contains %d quadrilaterals satisfying target monitor %s constraint %s', len(self.quadrilaterals[constraint]), constraint, str(self.constraints[constraint]))

    @multi_method.procedure (apply, lambda *args, **kwargs : args[2] == 'gradient')
    def apply (self, image, constraint):
        self.quadrilaterals[constraint] = list()
        # edge detection is performed on a Grayscale version of `image`
        grayscale_data = image.data if not image.isColor else cv2.cvtColor (image.data, cv2.COLOR_BGR2GRAY)
        assert grayscale_data is not None, 'no gray-scale image to analyze for Target monitor detection'
##        blurred_image  = cv2.GaussianBlur (grayscale_data, (5, 5), 0)
        blurred_image  = grayscale_data## <-------------------------------------
        min_edge       = self.constraints[constraint].minimum
        max_edge       = self.constraints[constraint].maximum
        for gray in cv2.split (blurred_image):
            for limit in xrange(0, 255, 26):
                if limit > 0:
                    retval, BWimage = cv2.threshold (gray, limit, 255, cv2.THRESH_BINARY) # Black/White image elements are zero for elements <= `limit` and 255 otherwise
                else:
                    BWimage = cv2.Canny  (gray, min_edge, max_edge, apertureSize=5) # `BWimage` is a 2-d, numpy.ndarray (cam height, cam width) of ints in range (0,255)
                    BWimage = cv2.dilate (BWimage, None)
                contours, hierarchy = cv2.findContours (BWimage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    # compute `contour` perimeter ...
                    perimeter = cv2.arcLength    (contour, True) # True -> `contour` is closed
                    # determine an approximation for `contour` ...
                    deviation = 0.02 * perimeter # how much `contour` approximation may deviate from original
                    contour   = cv2.approxPolyDP (contour, deviation, True) # True -> `contour` is closed
                    self.quadrilaterals[constraint].append (contour) if self.is_quadrilateral (contour) else False
        if not self.quadrilaterals[constraint]: raise NoMonitor ('image contains no features satisfying target monitor edge {} constraint {}'.format (constraint, self.constraints[constraint]))
        self.log.debug ('image contains %d quadrilaterals satisfying target monitor edge %s constraint %s', len(self.quadrilaterals[constraint]), constraint, str(self.constraints[constraint]))

    def as_radians (self, angle):
        return angle * self.rads_per_deg

    @staticmethod
    def is_quadrilateral (contour):
        return len(contour) == 4 and cv2.isContourConvex (contour) # 4 is number of vertices in a quadrilateral

    @staticmethod
    def _angle_cos_ (p0, p1, p2):
        d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
        return abs (numpy.dot (d1, d2) / numpy.sqrt (numpy.dot (d1, d1) * numpy.dot (d2, d2)))

    def __str__ (self):
        return '{}({}, {})'.format (self.__class__.__name__, str(self.thresholds['area']), str(self.thresholds['gradient']))


class Identifier (HasTraits):

    # externally-managed parameters ...
    targets    = List (trait=Target, minlen=1, desc='list of Targets to be identified in `image`')
    thresholds = Dict (key_trait=String, value_trait=Ratio, desc='constraits on what constitutes an identified Target')
    # internally-managed parameters ...
    image      = Instance (Image , desc='image analyzed for the presence of `targets`')
    target     = Instance (Target, desc='`targets` element whose color signature is best represented in `image`')
    masks      = Dict     (key_trait=Target, value_trait=numpy.ndarray, desc='masks identifying target signatures in `image`')
    counts     = Dict     (key_trait=Target, value_trait=int          , desc='pixel counts satisfying target signatures in `image`')
    computed   = Dict     (key_trait=String, value_trait=float        , desc='computed ratios')
    log        = Instance (logging.Logger, factory=logging.getLogger, args=('Target Identifier',))

    @classmethod
    def create (_self_, targets, settings):
        section    = 'Target Identification'
        similarity = settings.getfloat (section,'similarity')
        # separation = settings.getfloat (section,'separation')
        thresholds = dict(
                        similarity=similarity) # minimum ratio of image pixels that must satisfy Target color signature constraints to be considered a match
                        # separation=separation) # minimum ratio of dissimilarity between top 2 color-matched targets -- increase to require more separation
        parameters = dict (zip (('targets','thresholds'),(targets,thresholds)))
        return _self_ (**parameters)

    def identify (self, image):
        assert self.targets          , 'set of targets to be identified is undefined'
        assert image.data is not None, 'image to be scanned for targets is undefined'
        assert image.isColor         , 'Target identification requires a color image'
        self.image      = image
        del self.target
        del self.masks
        del self.counts
        for target in self.targets:
            self.log.debug (' searching image for %s with %s color signature',target.name,target.signature.color)
            # set color-matching bounds from color signature of current `target`
            # -- note: user may change `target` color signature on-the-fly so settings must always be checked
            # -- note: cv2.inRange() expects bounds order to be BGR rather than RGB
            dimmest   = numpy.array ((target.signature.dimmest  .Blue, target.signature.dimmest  .Green, target.signature.dimmest  .Red), dtype=numpy.uint8)
            brightest = numpy.array ((target.signature.brightest.Blue, target.signature.brightest.Green, target.signature.brightest.Red), dtype=numpy.uint8)
            self.masks [target] = cv2.inRange (image.data, dimmest, brightest) # mask all image.data pixels not in [dimmest,brightest]
            self.counts[target] = cv2.countNonZero (self.masks[target])        # number of image.data pixels, in [dimmest,brightest], for target
        # determine top 2 pixel color match counts (number of image.data pixels, in [dimmest,brightest], for each target) ...
        targets = [target for (target,count) in sorted (self.counts.items(), key = lambda (target,count): count)] # ordered from least to most pixel matches in `image`
        target  = targets[-1] # [-1] -> target with best signature match
        counts  = self.counts[target]
        ratio   = float(counts)/image.size
        self.computed['similarity'] = ratio
        if ratio < self.thresholds['similarity']:
            raise NoTarget ('no target satisfies minimum similarity ratio of {:.2%} :: best match is {!s} at {:.2%}'.format (self.thresholds['similarity'], target, ratio))
        self.target  = target
        self.log.info (' {} identified in image :: {:.2%} of image pixels match its {} color signature'.format (self.target.name, ratio, self.target.signature.color))


class Transformer (HasTraits):
    """ a calculator of position based on a perspective transformation """

    # externally-managed parameters ...
    camera = Instance (Camera, desc='image provider')
    target = Instance (Target, desc='Target detected in `image`')
    # internally-managed parameters ...
    center   = Property (depends_on='camera', desc='camera center (pixels) as an array for perspective transformation calculation')
    monitor  = Property (depends_on='target', desc='vertices of the target device monitor as an array for perspective transformation calculation')
    vertices = Instance (Quadrilateral      , desc='vertices of identified Target monitor in image')
    position = Instance (Position           , desc='position `target` mouse pointer should be (moved to) based on `camera` angle')
    log      = Instance (logging.Logger, factory=logging.getLogger, args=('Transformer',))

    def process (self, target, vertices):
        """
        PURPOSE
            calculate mouse pointer position, on Target monitor, from Target monitor location on `camera` sensor
        PARAMETERS
            * vertices : vertices of Target monitor on `camera` sensor (a `Quadrilateral` instance)
        RETURNS
            * nothing
        NOTES
            * Upon (successful) completion of this procedure, `self.position` ia the `Position`
                instance the mouse pointer should be (moved to) on the Target monitor.
        """
        self.target   = target
        self.vertices = vertices
        self.position = None
        vertices      = numpy.float32 ([vertices.upperleft, vertices.upperright, vertices.lowerleft, vertices.lowerright]) # vertices orderd as expected by cv2.getPerspectiveTransform
        transform     = cv2.getPerspectiveTransform (vertices, self.monitor)
        xy            = cv2.perspectiveTransform    (self.center[None,None,:], transform)
        self.position = Position (x=int(xy[0,0,0]), y=int(xy[0,0,1]))
        self.log.debug (' computed position is: %s', str(self.position))

    @cached_property
    def _get_center (self):
        return numpy.float32 ([self.camera.center.x, self.camera.center.y])

    @cached_property
    def _get_monitor (self):
        # cv2.getPerspectiveTransform expects 32-bit floats ordered as follows: [upper left, upper right, lower left, lower right]
        return numpy.float32 ([[0,0],[self.target.monitor.width,0],[0,self.target.monitor.height],[self.target.monitor.width, self.target.monitor.height]])

    def __str__ (self):
        return '{}({}, {})'.format (self.__class__.__name__, str(self.camera), str(self.target))


class Transmitter (HasTraits):
    """ inter-device, data transmitter (Wise Ball to Target) """

    # externally-managed parameters ...
    timeout   = Float (default_value=0.5, desc='timeout on blocking socket operations (seconds) :: `None` disables timeouts on socket operations')
    target    = Instance (Target, desc='recipient of directive transmissions')
    directive = Directive
    # internally-managed parameters ...
    log = Instance (logging.Logger, factory=logging.getLogger, args=('Transmitter',))

    def process (self, target, directive):
        self.target    = target
        self.directive = directive
        self.log.debug (' preparing to transmit %s to %s', directive, str(target))
        try:
            pass
        except Exception as details:
            self.log.warning (' %s could not be encoded for transmission :: %s', str(directive), details)
        else:
            sender = socket.socket (socket.AF_INET, socket.SOCK_STREAM)
            sender.settimeout (self.timeout)
            try:
                _target_ = socket.gethostbyname (target.name)
            except socket.gaierror as details:
                self.log.warning (' %s not found :: %s', str(target), details)
            else:
                try:
                    sender.connect ((_target_, target.port))
                except socket.error as details:
                    self.log.warning (' unable to connect to %s  in %s seconds :: %s', str(target), sender.gettimeout(), details)
                else:
                    try:
                        sender.sendall (directive)
                        self.log.debug (' transmitted %s to %s', directive, str(target))
                    except socket.error as details:
                        self.log.warning (' unable to send %s to %s in %s seconds :: %s', directive, str(target), sender.gettimeout(), details)
                finally:
                    try:
                        sender.shutdown (socket.SHUT_RDWR)
                        sender.close()
                    except socket.error as details:
                        pass
                        #self.log.warning (' unable to cleanly close connection to %s :: %s', str(target), details)

    def __str__ (self):
        return '{}(timeout:{})'.format (self.__class__.__name__, self.timeout)


class WiseBall (HasTraits):

    # externally-managed parameters ...
    camera      = Instance (Camera     , desc='video images provider')
    identifier  = Instance (Identifier , desc='Target identifier in video images')
    detector    = Instance (Detector   , desc='Target monitor detector in video images')
    transformer = Instance (Transformer, desc='Target mouse pointer position computer')
    transmitter = Instance (Transmitter, desc='Target directives transmitter')
    # internally-managed parameters ...
    image   = Instance (Image        , desc='image analyzed for the presence of Targets')
    target  = Instance (Target       , desc='2 most-recently identified Targets')
    monitor = Instance (Quadrilateral, desc='corner points of target monitor in a video frame')
    button  = Button
    keyboard_mapping = {VK_UP: 'UP', VK_DOWN: 'DOWN'}
    log     = Instance (logging.Logger, factory=logging.getLogger, args=('Wise Ball',))

    @classmethod
    def create (_self_, settings):
        # configure Targets to be identified/directed by the Wise Ball ...
        sections = [this for this in settings.sections() if this.startswith('Target <:> ')]
        targets  = list()
        for section in sections:
            targets.append (Target.create (section, settings))
        for this, that in pairs_of (targets):
            assert this != that, ' {} is not unique'.format(this)
        # configure Wise Ball components ...
        camera      = Camera    .create (settings)          # video camera used to image Targets
        identifier  = Identifier.create (targets, settings) # target device identifier
        detector    = Detector  .create (settings)          # detector of Target display monitor in video frames
        transformer = Transformer (camera=camera)           # position calculator of the target device mouse pointer
        transmitter = Transmitter ()                        # transmit mouse button,pointer  and display directives to Target
        parameters  = dict (zip (('camera', 'identifier', 'detector', 'transformer', 'transmitter'),(camera, identifier, detector, transformer, transmitter)))
        return _self_ (**parameters)

    def start (self):
        self.camera.connect()

    def stop (self):
        self.camera.disconnect()

    def operate (self):
        self.get_frame       ()
        self.identify_target ()
        self.locate_monitor  ()
        self.direct_mouse    ()

    def get_frame (self):
        self.camera.capture()
        self.image = self.camera.frame

    def identify_target (self):
        try:
            self.identifier.identify (self.image)
        except NoTarget as details:
            if not self.target: raise NoTarget (str(details))
        else:
            if not self.target or self.target != self.identifier.target:
                if self.target:
                    self.log.debug (' notifying %s to display its identity signature since it is no longer the target', str(self.target))
                    self.transmitter.process (self.target, SHOW_SIGNATURE)
                self.target = self.identifier.target
                self.log.debug (' notifying %s to hide its identity signature since it is now the target', str(self.target))
                self.transmitter.process (self.target, WITHDRAW_SIGNATURE)
        assert self.target is not None, 'LOGIC ERROR : no identified target upon nominal procedure ({}) completion'.format (self.identify_target.func_name)

    def locate_monitor (self):
        self.detector.process (self.image)
        self.monitor = Quadrilateral.create (self.detector.quadrilaterals['area'][-1]) # [-1] -> choose feature with largest area

    def direct_mouse (self):
        assert self.target,  'target directives not transmitted -- target is unidentified'
        assert self.monitor, 'target directives not transmitted -- target monitor not located'
        self.transformer.process (self.target, self.monitor)
        keystrokes  = [key for key in (VK_UP,VK_DOWN) if Keyboard.pressed(key)]
        keystroke   = keystrokes[-1] if keystrokes else None
        self.button = self.keyboard_mapping[keystroke] if keystroke else self.button
        directive   = MOUSE_DIRECTIVE.format (button=self.button, position=self.transformer.position)
        self.transmitter.process (self.target, directive)

    def __del__  (self):
        self.stop()


class Displays (HasTraits):


    # externally-managed parameters ...
    pointer = Instance (WiseBall, desc='Wise Ball pointer whose internal state is to be displayed')
    # internally-managed parameters ...
    windows = dict() # mapping of components to their window names
    Green   = (0,192,0) # (B,G,R)
    Red     = (0,0,128) # (B,G,R)
    font    = dict (fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(0,128,0), thickness=2, lineType=cv2.CV_AA) # font specifications for text annotations
    log     = Instance (logging.Logger, factory=logging.getLogger, args=('Displays',))

    def setup (self):
        self.components = self.pointer.identifier, self.pointer.detector
        # create a window for each component ...
        windows         = 'Target Identification', 'Target Monitor Detection'
        self.windows.update (zip(self.components, windows))
        [cv2.namedWindow (this) for this in self.windows.values()]
        # add track bars to the "Target Identification" window to adjust identifier.thresholds ...
        identifier = self.pointer.identifier
        window     = self.windows[identifier]
        cv2.createTrackbar ('similarity', window, 10, 20, lambda x: identifier.thresholds.update (dict (similarity=0.001+0.01*x))) # provides range [0.001, 0.4] :: default is 0.2
        # add track bars to the "Target Monitor Detection" window to adjust detector.constraints ...
        detector = self.pointer.detector
        window   = self.windows[detector]
        cv2.createTrackbar ('min area', window, 5, 10, lambda x: setattr (detector.constraints['area'    ], 'minimum',      x)    ) # provides range [ 0,10] percent :: default is  5%
        cv2.createTrackbar ('max area', window, 5, 10, lambda x: setattr (detector.constraints['area'    ], 'maximum', 10+8*x)    ) # provides range [10,90] percent :: default is 55%
        cv2.createTrackbar ('angle'   , window, 3, 10, lambda x: setattr (detector.constraints.update((('angle',3*x),)))          ) # provides range [0,30] degrees :: default is 9
        cv2.createTrackbar ('min edge', window, 5, 10, lambda x: setattr (detector.constraints['gradient'], 'minimum',  500+450*x)) # provides range [ 500,  5000] counts/pixel :: default is 2300
        cv2.createTrackbar ('max edge', window, 5, 10, lambda x: setattr (detector.constraints['gradient'], 'maximum', 5000+500*x)) # provides range [5000, 10000] counts/pixel :: default is 7500

    def update (self):
        [self.refresh (component) for component in self.components]
        cv2.waitKey (1) # milli-seconds :: required to signal cv2 to actually display the image :: 0 signals an indefinite wait

    refresh = multi_method.dispatcher()

    @multi_method.procedure (refresh, lambda *args, **kwargs : isinstance (args[1], Identifier))
    def refresh (self, identifier):
        if identifier.image is None or identifier.image.data is None : return
        window = self.windows[identifier]
        # apply the identifier mask to the image ...
        mask = cv2.bitwise_and (identifier.image.data, identifier.image.data, mask=identifier.masks[identifier.target]) if identifier.target else numpy.copy (identifier.image.data)
        # render the image mask and annotations ...
        annotations = list()
        if identifier.target:
            annotations.append ('{} similarity threshold minimum is {:.2%} : actual is {:.2%}'.format(identifier.target.signature.color, identifier.thresholds['similarity'], identifier.computed['similarity']))
            annotations.append ('{}'.format(identifier.target.name))
        else:
            annotations.append ('similarity threshold minimum is {:.2%} : actual is {:.2%}'.format(identifier.thresholds['similarity'], identifier.computed['similarity']))
            annotations.append ('no target identified')
        emphasis = 1,1,2 # font thickness paired with `annotations`
        xoffset  = 10 # pixels
        self.font['color'    ] = self.Green if identifier.target else self.Red
        self.font['fontScale'] = 0.75
        for this, (annotation, thickness) in enumerate (zip(annotations,emphasis)):
            self.font['thickness'] = thickness
            yoffset = identifier.image.height - (this+1)*25
            cv2.putText (img=mask, text=annotation, org=(xoffset,yoffset), bottomLeftOrigin=False, **self.font)
        cv2.imshow  (window, mask)

    @multi_method.procedure (refresh, lambda *args, **kwargs : isinstance (args[1], Detector))
    def refresh (self, detector):
        if detector.image is None or detector.image.data is None : return
        image = numpy.copy (detector.image.data)
        # place messages in the display window ...
        annotations = list()
        annotations.append ('image contains {} quadrilaterals satisfying interior angle constraint of {} degrees'.format (len(detector.quadrilaterals['angle'   ]), str(detector.constraints['angle'   ])))
        annotations.append ('image contains {} quadrilaterals satisfying area percentage constraint {}'          .format (len(detector.quadrilaterals['area'    ]), str(detector.constraints['area'    ])))
        annotations.append ('image contains {} quadrilaterals satisfying edge gradient   constraint {}'          .format (len(detector.quadrilaterals['gradient']), str(detector.constraints['gradient'])))
        annotations.append ('largest feature is {:.0f}% of the image area'.format (detector.areas[-1])) if detector.quadrilaterals['area'] else False
        self.font['color'    ] = self.Green if detector.quadrilaterals['area'] else self.Red
        self.font['fontScale'] = 0.6
        self.font['thickness'] = 2
        xoffset = 10
        [cv2.putText (img=image, text=annotation, org=(xoffset, detector.image.height-(this+1)*20), bottomLeftOrigin=False, **self.font) for this, annotation in enumerate (annotations)]
        # outline target monitor candidates in the image ...
        cv2.drawContours (image, detector.quadrilaterals['area'][ -1], -1, self.Green, 15) if detector.quadrilaterals['area'] else False # Green circles denoting most-likely, target monitor vertices
        cv2.drawContours (image, detector.quadrilaterals['area']     , -1, self.Red  ,  1) # Red rectangles denoting quadrilaterals satisfying all constraints
        cv2.imshow (self.windows[detector], image)

    def close (self):
        cv2.destroyAllWindows()

    def __del__  (self):
        self.close()


if __name__ == '__main__':
    '''
    PURPOSE
        provide a command-line script that demonstrates the following capabilities of a Wise Ball
        pointer:
            - Target devices identification
            - Target devices monitor detection
            - Target devices mouse pointer positioning and mouse button control
    NOTES
        * Execute one of the following commands:
                > python WiseBall.py --help
                > python WiseBall.py -h
            from a shell window to display a brief description of the command-line parameters.
    '''

    # standard Python packages ...
    from   argparse     import ArgumentParser
    from   ConfigParser import SafeConfigParser
    # generic extensions to standard Python ...
    # ... none
    # application-specific Python modules ...
    # ... none

    # initialize the commandline arguments parser ...
    parser = ArgumentParser()
    parser.add_argument ('--verbosity' , type=str, default='low'         , help='logging verbosity', choices=('low','medium','high'))
    parser.add_argument ('--configfile', type=str, default='WiseBall.ini', help='configuration file name')
    parser.add_argument ('--stopkey'   , type=str, default='Esc'         , help='STOP execution key')
    arguments = parser.parse_args()
    stopkey   = VK_ESCAPE if arguments.stopkey == 'Esc' else ord (arguments.stopkey)
    verbosity = dict (low=logging.WARN, medium=logging.INFO, high=logging.DEBUG)
    logging.basicConfig (level=verbosity[arguments.verbosity])
    # logging.basicConfig (filename='WiseBall.log', filemode='w', level=verbosity[arguments.verbosity])
    log = logging.getLogger ('Simulator')
    log.info (' started')
    try:
        configuration = SafeConfigParser()
        configuration.readfp (open(arguments.configfile))
        pointer  = WiseBall.create (configuration)
        displays = Displays (pointer=pointer)
        pointer.start()
        displays.setup()
    except (AssertionError, NoCamera) as details:
        log.error (' %s',details)
    else:
        while not Keyboard.pressed (stopkey):
            try:
                pointer.operate()
            except (AssertionError, NoImage, NoTarget, NoMonitor) as details:
                log.warning (' %s',details)
            except NoCamera as details:
                camera.connect()
            except Exception as details:
                log.exception (details)
                break
            else:
                log.info (' directing %s', pointer.target)
            displays.update()
        displays.close()
        pointer.stop()
    finally:
        log.info (' stopped')
        logging.shutdown()
