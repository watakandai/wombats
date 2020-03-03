# local packages
from wombats.factory.object_factory import ObjectFactory
from wombats.automaton.pdfa import PDFABuilder
from wombats.automaton.fdfa import FDFABuilder


class AutomatonCollection(ObjectFactory):
    """
    registering the builders for the different types of automaton objects
    with a more readable interface to our generic factory class
    """

    def get(self, automaton_type, **kwargs):
        """
        allows for more readble creation / access to a concrete
        wombats.automaton objects

        :param      automaton_type:  The automaton type
        :type       automaton_type:  string
        :param      kwargs:          The keywords arguments to pass to the
                                     specific automaton constructor
        :type       kwargs:          dictionary

        :returns:   an intialized / active reference to the desired type of
                    automaton object
        :rtype:     { return_type_description }
        """

        return self.create(automaton_type, **kwargs)


activeAutomaton = AutomatonCollection()
activeAutomaton.register_builder('PDFA', PDFABuilder())
activeAutomaton.register_builder('FDFA', FDFABuilder())
