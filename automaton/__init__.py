from wombats.automaton.pdfa import PDFABuilder
from wombats.automaton.fdfa import FDFABuilder
from wombats.automaton.factory import AutomatonCollection

activeAutomaton = AutomatonCollection()
activeAutomaton.register_builder('PDFA', PDFABuilder())
activeAutomaton.register_builder('FDFA', FDFABuilder())
