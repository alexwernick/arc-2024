from arc_2024.inductive_logic_programming.first_order_logic import (
    ArgType,
    Literal,
    Predicate,
    Variable,
)
from arc_2024.inductive_logic_programming.FOIL import FOIL


def test_FOIL():
    """
    Test FOIL algorithm
    This example is taken from the paper
    'Lerning Logical Definitions from Relations' by Quinlan et al.
    The example is about learning a 'can-reach' from 'linked-to' realtions.
    """
    node_type = ArgType("node", [0, 1, 2, 3, 4, 5, 6, 7, 8])  # 9 nodes

    target_predicate = Predicate("can-reach", 2, [node_type, node_type])

    V1 = Variable("V1", node_type)
    V2 = Variable("V2", node_type)

    target_literal = Literal(predicate=target_predicate, args=[V1, V2])

    # Available predicates
    predicates = [Predicate("linked-to", 2, [node_type, node_type])]

    # Positive examples
    positive_examples = {
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (0, 6),
        (0, 8),
        (1, 2),
        (3, 2),
        (3, 4),
        (3, 5),
        (3, 6),
        (3, 8),
        (4, 5),
        (4, 6),
        (4, 8),
        (6, 8),
        (7, 6),
        (7, 8),
    }
    examples = [
        ((i, j) in positive_examples, {V1.name: i, V2.name: j})
        for i in range(10)
        for j in range(10)
        if i != j
    ]

    # Background facts for predicates
    background_knowledge = {
        "linked-to": {
            (0, 1),
            (0, 3),
            (1, 2),
            (3, 2),
            (3, 4),
            (4, 5),
            (4, 6),
            (6, 8),
            (7, 6),
            (7, 8),
        }
    }

    foil = FOIL(target_literal, predicates, background_knowledge)
    foil.fit(examples)

    # rules = foil.rules
