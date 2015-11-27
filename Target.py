'''
CONTENTS
    components implementing a Target device interface with a Wise Ball

NOTES
    * This module must be executing on the Target machine to be directed by a Wise Ball.
    * This module may be imported (e.g., for unit testing) or run as a script from a command line.
    * If run as a command-line script (see __main__ at the bottom), it provides an interface to
        receive and respond to Wise Ball directives.

AUTHOR
    jgv.home@gmail.com
'''


# standard Python packages ...
import logging, os, socket, Tkinter
from   threading import Thread
# generic extensions to standard Python ...
import pyautogui                                    # https://pyautogui.readthedocs.org/en/latest/ -- mouse control
from   traits.api import Constant, HasTraits, Instance, Property, Range, String # http://www.lfd.uci.edu/~gohlke/pythonlibs/#ets
from   traits.api import cached_property, undefined
# application-specific Python modules ...
from   utils import SHOW_SIGNATURE, WITHDRAW_SIGNATURE, Directive, Mouse, multi_method


DEVICE = socket.gethostname()

class NoTransmission      (Exception) : pass
class PartialTransmission (Exception) : pass


class Monitor (HasTraits):
    """ Target device monitor screen """

    width  = Constant (value=pyautogui.size()[0], desc='monitor width (pixels)')
    height = Constant (value=pyautogui.size()[1], desc='monitor width (pixels)')

    def __str__ (self):
        return '{}(width:{}, height:{})'.format (self.__class__.__name__, self.width, self.height)


class Receiver (HasTraits):
    """ Target device component that receives Wise Ball Directives """

    # externally-specified parameters ...
    port    = Range (low=3000, high=5000, value=undefined, exclude_low=False, exclude_high=False, desc='socket communications port number')
    timeout = Range (low=0, high=10, value=3.0, exclude_low=False, exclude_high=False, desc='max time (secs) for completion of socket communique -- `None` (as long as needed) is typical for operational use')
    # internally-managed parameters ...
    listener     = Instance (socket._socketobject, desc='monitor of contacts on `port`')
    transmission = Directive
    log          = Instance (logging.Logger, factory=logging.getLogger, args=('Receiver',))

    def start (self):
        self.listener = socket.socket (socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind ((DEVICE, self.port)) # host='localhost' does not work -- at least when the transmitter is on the same machine
        self.listener.settimeout (self.timeout)
        self.listener.listen(5) # await sender contacts (5 is backlog limit)
        self.log.info (' awaiting contacts on port %d', self.port)

    def stop (self):
        if not self.listener : return
        try:
            # self.listener.shutdown (socket.SHUT_RDWR) ... raises: socket error [Errno 10057] A request to send or receive data was disallowed ...
            self.listener.close()
        except socket.error as details:
            self.log.info (' did not cleanly close its listener :: %s', details)
        del self.listener
        self.log.info (' stopped')

    def receive (self):
        """
        PURPOSE
            receive a single data transmission from another (Wise Ball or Target) device
        PARAMETERS
            * none
        RETURNS
            * nothing
        SIDE-EFFECTS
            * `self.transmission` is reset.
            * Messages may be logged.
        NOTES
            * This method may be used inside a (threaded) loop in order to receive on-demand transmissions.
            * See: http://code.activestate.com/recipes/408859-socketrecv-three-ways-to-turn-it-into-recvall/
        """
        del self.transmission
        alldata = list()
        try:
            receiver, sender = self.listener.accept()
        except socket.timeout as details:
            self.log.debug (' no contacts in %.1f seconds', self.timeout)
        else:
            self.log.debug (' contacted by %s', sender)
            receiver.settimeout (self.timeout)
            while True:
                try:
                    data = receiver.recv (1024)
                    self.log.debug (' received <%s> character transmission from %s', data, sender)
                    if not data: break
                    alldata.append (data)
                except socket.timeout as details:
                    raise PartialTransmission (' data transmission, from {}, not completed in {} seconds'.format(sender,self.timeout))
            try:
                receiver.shutdown (socket.SHUT_RDWR)
                receiver.close()
            except socket.error as details:
                self.log.warning (' did not cleanly close connection with %s :: %s', sender,details)
        if not alldata : raise NoTransmission ('no transmission received in {} seconds'.format(self.timeout))
        self.transmission = ''.join(alldata)

    def __del__ (self):
        self.stop()

    def __str__ (self):
        return '{0}(port={1},timeout={2})'.format (self.__class__.__name__, self.port, self.timeout)


class Signature (HasTraits):

    color   = String (desc='color name')
    width   = Property (depends_on='image', desc='image width (pixels)')
    height  = Property (depends_on='image', desc='image height (pixels)')
    display = Instance (Tkinter.Tk        , desc='graphics rendering engine')
    thread  = Instance (Thread            , desc='`display` thread showing `image`')
    # internally-managed parameters ...
    log     = log  = Instance (logging.Logger, factory=logging.getLogger, args=('Signature',))

    @staticmethod
    def is_relevant (directive):
        return isinstance (directive, str) and directive in (SHOW_SIGNATURE, WITHDRAW_SIGNATURE)

    @staticmethod
    def is_show (directive):
        return isinstance (directive, str) and directive == SHOW_SIGNATURE

    @staticmethod
    def is_withdraw (directive):
        return isinstance (directive, str) and directive == WITHDRAW_SIGNATURE

    process = multi_method.dispatcher()

    @multi_method.procedure (process, lambda *args, **kwargs : args[0].is_show(args[1]))
    def process (self, directive):
        self.show()

    @multi_method.procedure (process, lambda *args, **kwargs : args[0].is_withdraw(args[1]))
    def process (self, directive):
        self.remove()

    @multi_method.procedure (process, lambda *args, **kwargs : not args[0].is_relevant(args[1]))
    def process (self, directive):
        self.log.debug (' ignoring <%s> directive :: not relevant', str(directive))

    show = multi_method.dispatcher()

    @multi_method.procedure (show, lambda *args, **kwargs : not args[0].display)
    def show (self):
        self.log.debug (' processing request to show %s', self)
        self.display = Tkinter.Tk()
        self.display.overrideredirect (True)
        self.display.title ('{} Identification Signature'.format(DEVICE))
        self.display['bg'] = self.color
        width, height = self.display.winfo_screenwidth(), self.display.winfo_screenheight()
        self.display.geometry ("%dx%d+0+0" % (width, height))
        self.display.focus_set()
        #self.display.bind ('<Escape>', lambda event: event.widget.quit())
        self.thread = Thread (target=self.display.mainloop, name='{} ID Signature'.format(DEVICE))
        self.thread.start()

    @multi_method.procedure (show, lambda *args, **kwargs : args[0].display)
    def show (self):
        self.log.debug (' request to show %s ignored :: presently being displayed', self)

    remove = multi_method.dispatcher()

    @multi_method.procedure (remove, lambda *args, **kwargs : args[0].display)
    def remove (self):
        self.log.debug (' processing request to remove %s', self)
        try:
            self.display.destroy()
        except Tkinter.TclError as details:
            if not str(details).endswith('application has been destroyed') : self.log.warning (' closing window :: %s',details)
        del self.display
        del self.thread

    @multi_method.procedure (remove, lambda *args, **kwargs : not args[0].display)
    def remove (self):
        self.log.debug (' request to remove %s ignored :: not presently being displayed', self)

    @cached_property
    def _get_width (self):
        return self.image.width()

    @cached_property
    def _get_height (self):
        return self.image.height()

    def __del__ (self):
        self.remove()

    def __str__ (self):
        return '{}(name:{})'.format (self.__class__.__name__, self.color)


class Target (HasTraits):

    # externally-specified parameters ...
    name      = Constant (value=DEVICE, desc='target machine name on local network')
    signature = Instance (Signature, desc='Target color signature for identification by Wise Ball')
    monitor   = Instance (Monitor  , desc='Target screen')
    receiver  = Instance (Receiver , desc='receiver of Wise Ball directives')
    directive = Directive
    # internally-managed parameters ...
    mouse     = Instance (Mouse         , factory=Mouse            , args=())
    log       = Instance (logging.Logger, factory=logging.getLogger, args=(name.default_value,))

    @classmethod
    def create (_self_, settings):
        section    = 'Target <:> {}'.format(DEVICE) # [Section] name suffix (after <:>) assumed to be the machine name on local network
        assert settings.has_section (section), '{} Section not found in configuration file'.format(section)
        port       = settings.getint (section, 'port')
        color      = settings.get    (section, 'color')
        signature  = Signature (color=color) # creating `image` instance prior to Tkinter.Tk instance creation raises: "RuntimeError: Too early to create image"
        monitor    = Monitor   ()
        receiver   = Receiver  (port=port)
        parameters = dict (zip (('signature','monitor','receiver'),(signature,monitor,receiver)))
        return _self_ (**parameters)

    def start (self):
        self.receiver.start()
        self.signature.show()

    def stop (self):
        self.receiver.stop()
        self.signature.remove()

    def execute (self):
        try:
            self.receiver.receive()
        except NoTransmission as details:
            self.signature.show()
        else:
            self.directive = self.receiver.transmission
            if self.mouse.is_relevant (self.directive):
                self.signature.remove() # never display signature if mouse directives are being received
                self.mouse.process (self.directive)
            else:
                self.signature.process (self.directive)

    def __str__ (self):
        return '{}({}, {})'.format(self.__class__.__name__, self.name, str(self.signature))


if __name__ == '__main__':
    '''
    PURPOSE
        present an interface to receive and respond to the following Wise Ball directives:
            - mouse pointer positioning
            - mouse button press/release
            - color signature display/withdraw (facilitate identification by a Wise Ball)
    NOTES
        * Execute one of the following commands:
                > python Target.py --help
                > python Target.py -h
            from a shell window to display a brief description of the command-line parameters.
        * If not deployed on a Microsoft OS, with win32con (http://sourceforge.net/projects/pywin32/files/pywin32/) installed,
            script termination from the keyboard is disabled.  OpenCV (cv2) has a cross-platform keyboard interface, but a
            requirement is to avoid installation of OpenCV on target machines.
    '''

    # standard Python packages ...
    from   argparse     import ArgumentParser
    from   ConfigParser import SafeConfigParser
    import sys
    # generic extensions to standard Python ...
    if sys.platform == 'win32': from win32con import VK_ESCAPE # [MS Windows Specific] [Esc] key -- VK is virtual key :: http://sourceforge.net/projects/pywin32/files/pywin32/
    # application-specific Python modules ...
    from   utils import Keyboard # [MS Windows-specific]

    parser = ArgumentParser() # commandline arguments parser
    parser.add_argument ('--verbosity' , type=str, default='low'         , help='logging verbosity', choices=('low','medium','high'))
    parser.add_argument ('--configfile', type=str, default='WiseBall.ini', help='configuration file name')
    parser.add_argument ('--stopkey'   , type=str, default='Esc'         , help='STOP execution key')
    arguments = parser.parse_args()
    stopkey   = VK_ESCAPE if arguments.stopkey == 'Esc' else ord (arguments.stopkey)
    verbosity = dict (low=logging.WARN, medium=logging.INFO, high=logging.DEBUG)
    logging.basicConfig (level=verbosity[arguments.verbosity])
    # logging.basicConfig (filename='{}.log'.format(DEVICE), filemode='w', verbosity[arguments.verbosity])
    log = logging.getLogger (DEVICE)
    log.info (' started')
    try:
        configuration = SafeConfigParser()
        configuration.readfp (open(arguments.configfile))
        target = Target.create (configuration)
        target.start()
    except (AssertionError,) as details:
        log.error (' %s',details)
    else:
        while not Keyboard.pressed (stopkey):
            try:
                target.execute()
            except (AssertionError, PartialTransmission) as details:
                log.warning (' %s',details)
            else:
                pass # TBD log.info (## ???)
        target.stop()
    finally:
        log.info (' stopped')
        logging.shutdown()
