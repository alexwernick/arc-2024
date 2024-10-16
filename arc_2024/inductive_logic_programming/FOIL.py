import copy
import itertools
import math
from typing import Any, NamedTuple

from arc_2024.inductive_logic_programming.first_order_logic import (
    ArgType,
    Clause,
    Constant,
    Literal,
    Predicate,
    Variable,
    evaluate_literal,
)


class FOIL:
    target_literal: Literal
    predicates: list[Predicate]
    background_knowledge: dict[str, set[tuple]]
    allow_recursion: bool
    beam_width: int
    _max_clause_length: int
    rules: list[Clause]
    _non_extendable_types: set[ArgType]

    class BeamItem(NamedTuple):
        clause: Clause
        old_positive_examples: list[dict[str, Any]]
        old_negative_examples: list[dict[str, Any]]
        non_extended_pos_covered: list[dict[str, Any]]

    class BestLiteral(NamedTuple):
        gain: float
        literal: Literal
        pos_covered: list[dict[str, Any]]
        neg_covered: list[dict[str, Any]]
        non_extended_pos_covered: list[dict[str, Any]]

    def __init__(
        self,
        target_literal: Literal,
        predicates: list[Predicate],
        background_knowledge: dict[str, set[tuple]],
        allow_recursion: bool = False,
        beam_width: int = 1,
        max_clause_length: int = 7,
        non_extendable_types=set(),
    ):
        """
        Initialize the FOIL algorithm.
        :param target_literal: The target literal to learn rules for.
        :param predicates: A list of Predicate objects.
        :param background_knowledge: A dict mapping predicate names to sets of facts.
        :param allow_recursion: If True, allow recursion in the learned rules.
        """
        self.target_literal = target_literal
        self.predicates = predicates
        self.background_knowledge = background_knowledge
        self.allow_recursion = allow_recursion
        self.beam_width = beam_width
        self._max_clause_length = max_clause_length
        self._non_extendable_types = non_extendable_types
        self.rules = []

    def fit(self, examples: list[tuple[bool, dict[str, Any]]]):
        """
        Learn rules to cover positive examples.
        :param examples: A list of tuples (is_positive, example)
        """
        # Separate positive and negative examples
        pos_examples: list[dict[str, Any]] = [ex for is_pos, ex in examples if is_pos]
        neg_examples: list[dict[str, Any]] = [
            ex for is_pos, ex in examples if not is_pos
        ]

        # Maybe try limiting number of shapes in clause?....
        uncovered_pos = pos_examples.copy()

        while uncovered_pos:
            # Learn a new clause to cover positive examples
            clause, best_pos_covered = self._new_clause(uncovered_pos, neg_examples)
            # Remove positive examples covered by this clause
            uncovered_pos = [d for d in uncovered_pos if d not in best_pos_covered]
            self.rules.append(clause)

            if self.allow_recursion:
                pass  # I think we need to add the clause to the background knowledge

    def predict(self, example: dict[str, Any]) -> bool:
        """
        Predict if the example is positive based on learned rules.
        """
        for rule in self.rules:
            if rule.covers([example], self.background_knowledge):
                return True
        return False

    def _new_clause(
        self,
        uncovered_pos_examples: list[dict[str, Any]],
        neg_examples: list[dict[str, Any]],
    ) -> tuple[Clause, list[dict[str, Any]]]:
        """
        Learn a new clause to cover positive examples.
        :param uncovered_pos_examples: A list of positive examples not yet covered.
        :param neg_examples: A list of negative examples.
        :return: A new Clause object.
        """
        old_positive_examples = copy.deepcopy(uncovered_pos_examples)
        old_negative_examples = copy.deepcopy(neg_examples)

        # Start with a beam with a clause with only the head (the target literal)
        beam: list[FOIL.BeamItem] = [
            FOIL.BeamItem(
                clause=Clause(self.target_literal),
                old_positive_examples=old_positive_examples,
                old_negative_examples=old_negative_examples,
                non_extended_pos_covered=old_positive_examples,
            )
        ]

        while not self._beam_contains_item_with_no_negatives(
            beam
        ):  # while negative examples are covered
            new_beam: list[FOIL.BeamItem] = []

            # We check clause length. Fine to just check first
            # item as they all have the same length
            if len(beam[0].clause.body) >= self._max_clause_length:
                raise Exception(
                    f"Could not find a valid clause: max_clause_length {self._max_clause_length} reached"  # noqa: E501
                )

            for beam_item in beam:
                best_literals = self._find_next_best_literals(
                    beam_item.clause,
                    uncovered_pos_examples,
                    beam_item.old_positive_examples,
                    beam_item.old_negative_examples,
                )

                for best_literal in best_literals:
                    clause = copy.deepcopy(beam_item.clause)
                    clause.add_literal(best_literal.literal)
                    new_beam.append(
                        FOIL.BeamItem(
                            clause=clause,
                            old_positive_examples=best_literal.pos_covered,
                            old_negative_examples=best_literal.neg_covered,
                            non_extended_pos_covered=best_literal.non_extended_pos_covered,  # noqa: E501
                        )
                    )

            if len(new_beam) == 0:
                raise Exception("Could not find a valid clause: no new literals found")

            # sort literals info gain (descending)
            new_beam = sorted(
                new_beam,
                key=lambda beam_item: self._information_gain(
                    len(beam_item.non_extended_pos_covered),
                    len(beam_item.old_positive_examples),
                    len(beam_item.old_negative_examples),
                    len(uncovered_pos_examples),
                    len(neg_examples),
                ),
                reverse=True,
            )

            # take top beam_width items
            beam = new_beam[: self.beam_width]

        # We may have more than one beam with no negative examples
        # so we take the one with the most positive examples
        beam = [
            beam_item for beam_item in beam if len(beam_item.old_negative_examples) == 0
        ]
        beam = sorted(
            beam,
            key=lambda beam_item: len(beam_item.old_positive_examples),
            reverse=True,
        )

        return (beam[0].clause, beam[0].non_extended_pos_covered)

    def _find_next_best_literals(
        self,
        clause: Clause,
        uncovered_pos_examples: list[dict[str, Any]],
        old_positive_examples: list[dict[str, Any]],
        old_negative_examples: list[dict[str, Any]],
    ) -> list[BestLiteral]:
        candidate_literals: list[Literal] = self._new_literals(clause)
        best_gain: float = 0.0
        best_literals: list[FOIL.BestLiteral] = []

        # Evaluate each candidate literal
        for literal in candidate_literals:
            if literal in clause.body:
                continue  # Avoid cycles

            if literal == clause.head:
                continue  # Avoid cycles

            new_positive_examples: list[dict[str, Any]] = []
            for example in old_positive_examples:
                new_positive_examples.extend(self._extend_example(example, literal))

            new_non_extended_positive_examples = self._un_extend_examples(
                uncovered_pos_examples, new_positive_examples
            )

            new_non_extended_positive_examples_count = len(
                new_non_extended_positive_examples
            )
            new_positive_count = len(new_positive_examples)
            new_negative_count = 0  # for now we assume best case is 0
            old_positive_count = len(old_positive_examples)
            old_negative_count = len(old_negative_examples)

            if not new_positive_count:
                continue  # Must cover some positive examples

            # Now compute the maximum possible gain from substitutions
            # so we can prune. Max gain is assuming that we will have no
            # negative examples covered by the new literal
            max_gain = self._information_gain(
                new_non_extended_positive_examples_count,
                new_positive_count,
                0,
                old_positive_count,
                old_negative_count,
            )

            # If max_gain is less than best_gain, prune and
            # avoid extending negative examples
            if max_gain < best_gain:
                continue

            new_negative_examples: list[dict[str, Any]] = []
            for example in old_negative_examples:
                new_negative_examples.extend(self._extend_example(example, literal))

            new_negative_count = len(new_negative_examples)

            gain = self._information_gain(
                new_non_extended_positive_examples_count,
                new_positive_count,
                new_negative_count,
                old_positive_count,
                old_negative_count,
            )

            if gain > best_gain:
                # We append to our list of best literals
                best_literals.append(
                    FOIL.BestLiteral(
                        gain=gain,
                        literal=literal,
                        pos_covered=new_positive_examples,
                        neg_covered=new_negative_examples,
                        non_extended_pos_covered=new_non_extended_positive_examples,
                    )
                )

                # no point having more best literals than beam width
                if len(best_literals) > self.beam_width:
                    # Sort the list (ascending order)
                    best_literals = sorted(best_literals, key=lambda lit: lit.gain)
                    # Remove the worst
                    best_literals.pop(0)
                    # we then set gain below to be the
                    # worst value of the best literals (first item in list)
                    best_gain = best_literals[0].gain

        return best_literals

    def _extend_example(
        self, example: dict[str, Any], literal_to_add: Literal
    ) -> list[dict[str, Any]]:
        """
        Extend the example with the target predicate.
        """
        # example will be like (true, {V1: 1, V2: 2})
        # literal will be like Q(V1, V3)
        # need to loop over possible values
        # for V3 and add to examples if it satisfies the literal
        extended_examples: list[dict[str, Any]] = []
        new_vars: list[Variable] = []

        for arg in literal_to_add.args:
            # Check if the argument is a constant
            # if so append to args
            if isinstance(arg, Constant):
                continue

            # Check if the argument is a variable in the example
            # if so get value from example
            if isinstance(arg, Variable) and arg.name not in example:
                new_vars.append(arg)
                continue

        if len(new_vars) == 0:
            if evaluate_literal(literal_to_add, example, self.background_knowledge):
                return [example]  # no new variables to add
            return []

        if len(new_vars) > 1:
            raise ValueError("We should only have at most one new variable to add")

        new_var: Variable = new_vars[0]

        for value in new_var.arg_type.possible_values(example):
            extended_example = copy.deepcopy(example)
            extended_example[new_var.name] = value
            if evaluate_literal(
                literal_to_add, extended_example, self.background_knowledge
            ):
                extended_examples.append(extended_example)

        return extended_examples

    def _new_literals_for_predicate(
        self,
        predicate: Predicate,
        clause: Clause,
        allow_variable_extension: bool = True,
    ) -> list[Literal]:
        """
        Generate possible literals to add to a clause's body.
        :param rule: The current rule being specialized.
        :return: A list of Literal objects.
        """
        possible_literals: list[Literal] = []
        # get the current variables in the clause
        current_vars = list(clause.variables)
        # get the literals already used in the clause so we don't repeat
        used_literals = set(clause.body)

        # if isinstance(predicate, RuleBasedPredicate):
        #     # Rule based preds need bound args when
        #     # checking coverage
        #     allow_variable_extension = False

        valid_vars_per_arg: list[list[Variable]] = [[] for _ in range(predicate.arity)]

        for var in current_vars:
            for i, arg_type in enumerate(predicate.arg_types):
                if arg_type == var.arg_type:
                    valid_vars_per_arg[i].append(var)

        # Generate all combinations of existing variables
        var_combinations = [
            combo
            for combo in itertools.product(*valid_vars_per_arg)
            if len(set(combo)) == len(combo)  # avoid duplicates
        ]

        if allow_variable_extension:
            # Introduce new variables (e.g., V1, V2)
            # We create a single additional variable for each type of argument
            new_vars_per_arg: list[list[Variable]] = [
                [] for _ in range(predicate.arity)
            ]
            new_var_number = 1 + len(clause.variables)
            for i, arg_type in enumerate(predicate.arg_types):
                new_vars_per_arg[i].append(Variable(f"V{new_var_number}", arg_type))

            # append the combinations of the new variable with the other variables
            for i, new_vars in enumerate(new_vars_per_arg):
                if new_vars[0].arg_type in self._non_extendable_types:
                    continue

                new_vars_with_other_args_per_arg = (
                    valid_vars_per_arg[:i] + [new_vars] + valid_vars_per_arg[i + 1 :]
                )
                var_combinations.extend(
                    itertools.product(*new_vars_with_other_args_per_arg)
                )

        # Create literals for each variable combination
        for vars in var_combinations:
            variables = list(vars)
            literal = Literal(predicate, variables)
            # negated_literal = Literal(predicate, variables, negated=True)
            # Avoid adding duplicate literals
            if literal not in used_literals:
                possible_literals.append(literal)

            # Avoid adding duplicate literals
            # if negated_literal not in used_literals:
            #     possible_literals.append(negated_literal)

        return possible_literals

    def _new_literals(
        self, clause: Clause, allow_variable_extension: bool = True
    ) -> list[Literal]:
        """
        Generate new literals to add to the clause.
        4 types of literals can be generated:
        - Xj = Xk
        - Xj != Xk
        - Q(V1, V2,..., Vk)
        - not Q(V1, V2,..., Vk)

        Foil investigates entire search space with
        three significant qualifications:
        - The literal must contain one existing variable
        - If Q is the same as the relation P on the LHS of the clause,
        possible arguments are restricted to prevent recursion
        - The form of the Gain heursitic allows a kind of purning akin
        to aphla-beta pruning in
        """
        literals: list[Literal] = []
        for predicate in self.predicates:
            # Generate possible literals
            literals.extend(
                self._new_literals_for_predicate(
                    predicate, clause, allow_variable_extension
                )
            )

        return literals

        # TODO: extend to include inequality literals

    @staticmethod
    def _information_gain(
        new_non_extended_positive_examples_count: int,
        new_positive_count: int,
        new_negative_count: int,
        old_positive_count: int,
        old_negative_count: int,
    ) -> float:
        """
        Calculate the adjusted FOIL information gain when introducing new literals.
        """
        # Avoid division by zero and log of zero
        if (
            new_non_extended_positive_examples_count == 0
            or new_positive_count == 0
            or old_positive_count == 0
        ):
            return 0

        # Compute probabilities
        prob_old = old_positive_count / (old_positive_count + old_negative_count)
        prob_new = new_positive_count / (new_positive_count + new_negative_count)

        # Calculate information
        information_old = -math.log2(prob_old)
        information_new = -math.log2(prob_new)

        # Compute information gain
        gain = new_non_extended_positive_examples_count * (
            information_old - information_new
        )
        return gain

    @staticmethod
    def _un_extend_examples(
        old_positive_examples: list[dict[str, Any]],
        new_positive_examples: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        This slighly odd function is best exmplained by the following
        by look at the paper Leraning Logical Definitions
        from Relations by Quinlan et al.
        We are calculation the de extended examples in order to get T1++
        """
        is_pos_example_covered = True
        non_extended_positive_examples: list[dict[str, Any]] = []
        for old_example in old_positive_examples:
            for new_posivive in new_positive_examples:
                is_pos_example_covered = True
                for key in old_example.keys():
                    if old_example[key] != new_posivive[key]:
                        is_pos_example_covered = False
                        break
                if is_pos_example_covered:
                    break

            if is_pos_example_covered:
                non_extended_positive_examples.append(old_example)

        return non_extended_positive_examples

    def _beam_contains_item_with_no_negatives(
        self, beam: list["FOIL.BeamItem"]
    ) -> bool:
        for item in beam:
            if not item.old_negative_examples:
                return True
        return False
