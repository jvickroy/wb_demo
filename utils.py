'''
CONTENTS
    generic utility components

AUTHOR
    jgv.home@gmail.com
'''

# standard Python packages ...
from   collections import namedtuple
import logging, sys
# generic extensions to standard Python ...
import numpy     # http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
import pyautogui # https://pyautogui.readthedocs.org/en/latest/ -- mouse control
from   traits.api import Bool, Constant, Enum, HasTraits, Instance, Property, Range, String # http://www.lfd.uci.edu/~gohlke/pythonlibs/#ets
from   traits.api import cached_property
if sys.platform == 'win32': from win32api import GetAsyncKeyState # [MS Windows Specific] http://sourceforge.net/projects/pywin32/files/pywin32/


Position = namedtuple ('Position', field_names='x y')
Button   = Enum  ('UP', ('UP','DOWN'), desc='mouse button state')
Ratio    = Range (low=0.0, high=1.0, exclude_low=False, exclude_high=False)

# Wise Ball / Target communication directives ...
Directive          = String (minlen=1, desc='instructions transmitted by a Wise Ball to a Target')
SHOW_SIGNATURE     = 'present signature'
WITHDRAW_SIGNATURE = 'withdraw signature'
MOUSE_DIRECTIVE    = '[MOUSE] || button={button} || x={position.x} || y={position.y}' # template (filled by Wise Ball prior to transmission)


class multi_method (object):
    """
    PURPOSE
        provide parameter-based, dynamic-dispatch for Python functions and class methods
    USE
        foo = multi_method.dispatcher() # create a dispatcher for the `foo` procedure

        @multi_method.procedure (foo, lambda *args, **kwargs : <*args and/or **kwargs criteria for using this version of `foo` goes here>)
        def foo (*args, **kwargs):
            ... implementation for this version of foo

        @multi_method.procedure (foo, lambda *args, **kwargs : <*args and/or **kwargs criteria for using this version of `foo` goes here>)
        def foo (*args, **kwargs):
            ... implementation for this version of foo

        @multi_method.procedure (foo, lambda *args, **kwargs : <*args and/or **kwargs criteria for using this version of `foo` goes here>)
        def foo (*args, **kwargs):
            ... implementation for this version of foo

        bar = multi_method.dispatcher() # create a dispatcher for the `bar` procedure

        @multi_method.procedure (bar, lambda *args, **kwargs : <*args and/or **kwargs criteria for using this version of `bar` goes here>)
        def bar (*args, **kwargs):
            ... implementation for this version of bar

        @multi_method.procedure (bar, lambda *args, **kwargs : <*args and/or **kwargs criteria for using this version of `bar` goes here>)
        def bar (*args, **kwargs):
            ... implementation for this version of bar

        ... and so on ...

    REFERENCE
        "Clojure Style Multi Methods in Python"
            ( http://codeblog.dhananjaynene.com/2010/08/clojure-style-multi-methods-in-python/ )
    """

    @staticmethod
    def dispatcher():
        """
        Declares a multi-map based method which will switch to the
        appropriate function based on the results of the switcher function
        """
        def component (*args, **kwargs):
            for condition, func in component.functions:
                if condition (*args, **kwargs) : return func (*args,**kwargs)
            raise Exception ('no procedure defined for {0}'.format(args))
        component.functions = list()
        return component

    @staticmethod
    def procedure (dispatcher, condition):
        """
        The multi-method decorator which allows one method at a time
        to be added to the broader switch for the given result value
        """
        def inner (wrapped):
            dispatcher.functions.append ((condition, wrapped))
            # Change the method name from clashing with the multi and allowing
            # all multi methods to be written using the same name
            wrapped.func_name = '_' + wrapped.func_name + '_' + str(condition)
            return dispatcher
        return inner


def pairs_of (things):
    for i, this in enumerate (things[:-1]):
        for that in things[i+1:]:
            yield this, that


class Bounds (HasTraits):
    """  """

    minimum = Range (low=0.0, exclude_low=False)
    maximum = Range (low=0.0, exclude_low=True)

    def __str__ (self):
        return '(min:{}, max:{})'.format(self.minimum, self.maximum)


class Color (HasTraits):

    # externally-managed parameters ...
    min   = Constant (value=  0, desc='minimum pixel intensity value for any color')
    max   = Constant (value=255, desc='maximum pixel intensity value for any color')
    Red   = Range    (low=0, high=255, exclude_low=False, exclude_high=False, desc='Red pixel value')
    Green = Range    (low=0, high=255, exclude_low=False, exclude_high=False, desc='Green pixel value')
    Blue  = Range    (low=0, high=255, value=0, exclude_low=False, exclude_high=False, desc='Blue pixel value')
    # internally-managed parameters ...
    # none

    def __eq__ (self, other):
        return self.Red == other.Red and self.Green == other.Green and self.Blue == other.Blue

    def __ne__ (self, other):
        return self.Red != other.Red or self.Green != other.Green or self.Blue != other.Blue

    def __hash__ (self):
        return hash ((self.Red, self.Green, self.Blue))

    def __str__ (self):
        return 'RGB: ({},{},{})'.format(self.Red, self.Green, self.Blue)


class Keyboard (object):
    """ keyboard interface """

    @staticmethod
    def pressed (key=ord('X')):
        """
        PURPOSE
            determine whether or not keyboard key was pressed since the last call
        PARAMETERS
            * key : integer code for key to be checked
        RETURNS
            * `True` if pressed and `False` otherwise
        SIDE EFFECTS
            * none
        NOTES
            * none
        """
        result = GetAsyncKeyState (key) if sys.platform == 'win32' else None # `result` is short int
        return result


class Mouse (HasTraits):

    # externally-specified parameters ... none
    # internally-managed parameters ...
    pointer  = Instance (Position           , desc='Target mouse pointer calculated position')
    button   = Button
    log      = Instance (logging.Logger, factory=logging.getLogger, args=('Mouse',), transient=True) # transient=True -> do not serialize (pickle) this attribute
    pyautogui.FAILSAFE = False # disable Exception raising when pointer moves to upper-left screen corner -- useful escape during development

    @staticmethod
    def is_relevant (directive):
        return isinstance (directive, str) and directive.startswith('[MOUSE]')

    @staticmethod
    def parse (directive):
        try:
            tag, button, x, y = directive.split (' || ')
        except ValueError as details:
            raise BadDirective ('unrecognized mouse directive <{}>'.format(directive))
        keywords = '[MOUSE]', 'button=', 'x=', 'y='
        fields   =    tag   ,  button  ,  x  ,  y
        settings = list()
        for keyword, field  in zip (keywords, fields):
            assert field.startswith (keyword), 'directive <{}> contains an unrecognized or mispositioned field ({})'.format(directive,field)
            settings.append (field.replace(keyword,''))
        button, x, y = settings[1:] # ignore `tag`
        return str(button), int(x), int(y)

    process = multi_method.dispatcher()

    @multi_method.procedure (process, lambda *args, **kwargs : args[0].is_relevant(args[1]))
    def process (self, directive):
        self.log.debug (' processing <%s> directive', directive)
        self.button, x, y = self.parse (directive)
        self.pointer = Position (x,y)
        try:
            if self.button == 'UP':
                ##???pyautogui.mouseUp (button='left')
                pyautogui.moveTo (self.pointer.x, self.pointer.y)
            else:
                ##???pyautogui.mouseDown (button='left')
                pyautogui.mouseDown (button='left', x=self.pointer.x, y=self.pointer.y)
        except Exception as details:
            self.log.warning (' <%s> directive not processed :: %s', directive, details)

    @multi_method.procedure (process, lambda *args, **kwargs : not args[0].is_relevant(args[1]))
    def process (self, directive):
        self.log.debug (' ignoring <%s> directive :: not relevant', str(directive))

    def pointer_to (self, x, y):
        pyautogui.moveTo (x, y)

    def __str__ (self):
        return '{}({}, button:{})'.format(self.__class__.__name__, str(self.pointer), self.button)


class Percentage (HasTraits):
    """  """

    minimum = Range (low=0.0, high=100.0, value=  0.0, exclude_low=False, exclude_high=True )
    maximum = Range (low=0.0, high=100.0, value=100.0, exclude_low=True , exclude_high=False)

    def __str__ (self):
        return '(min:{}%, max:{}%)'.format(self.minimum, self.maximum)


class Quadrilateral (HasTraits):

    # externally-managed parameters ...
    upperleft  = Instance (Position, desc='upper left corner of quadrilateral')
    upperright = Instance (Position, desc='upper right corner of quadrilateral')
    lowerleft  = Instance (Position, desc='lower left corner of quadrilateral')
    lowerright = Instance (Position, desc='lower right corner of quadrilateral')
    # internally-managed parameters ...
    # none

    @classmethod
    def create (_self_, array):
        array = array.reshape (-1,2) if array.shape == (4,1,2) else array
        assert array.shape == (4,2), 'array does not contain Quadrilateral vertices :: shape is {}'.format (array.shape)
        return _self_ (upperleft=Position(*array[0]), lowerleft=Position(*array[1]), lowerright=Position(*array[2]), upperright=Position(*array[3]))

    def __contains__ (self, point):
        ## quick'n-dirty : make determination based on largest rectangle contained in `self`
        lefts  = [getattr (this, 'x') for this in self.upperleft, self.lowerleft]
        rights = [getattr (this, 'x') for this in self.lowerright, self.upperright]
        xmin, xmax = max (lefts), min (rights)
        if point.x < xmin or point.x > xmax : return False
        uppers = [getattr (this, 'y') for this in self.upperleft, self.upperright]
        lowers = [getattr (this, 'y') for this in self.lowerleft, self.lowerright]
        ymin, ymax = max (uppers), min (lowers)
        return ymin <= point.y <= ymax

    def __str__ (self):
        return '{0} (upper-left={1}, lower-left={2}, lower-right={3}, upper-right={4})'.format (self.__class__.__name__, self.upperleft, self.lowerleft, self.lowerright, self.upperright)

    def asarray (self):
        elements = [self.upperleft, self.lowerleft, self.lowerright, self.upperright]
        return numpy.array (elements)
