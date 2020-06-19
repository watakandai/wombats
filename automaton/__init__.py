from .pdfa import PDFABuilder
from .fdfa import FDFABuilder
from .transition_system import TSBuilder
from .factory import AutomatonCollection

active_automata = AutomatonCollection()
active_automata.register_builder('PDFA', PDFABuilder())
active_automata.register_builder('FDFA', FDFABuilder())
active_automata.register_builder('TS', TSBuilder())
