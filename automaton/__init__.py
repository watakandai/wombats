from wombats.automaton.pdfa import PDFABuilder
from wombats.automaton.fdfa import FDFABuilder
from wombats.automaton.factory import AutomatonCollection

active_automata = AutomatonCollection()
active_automata.register_builder('PDFA', PDFABuilder())
active_automata.register_builder('FDFA', FDFABuilder())
