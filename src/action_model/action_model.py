from pathlib import Path

from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain
from sam_learning.learners.sam_learning import SAMLearner

pddl_plus_domain: Domain = DomainParser(Path("../domains/blocks/blocks.pddl")).parse_domain()
