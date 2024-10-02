import copy
import itertools
import math
from typing import Any, Optional

from arc_2024.inductive_logic_programming.first_order_logic import (
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
    rules: list[Clause]

    def __init__(
        self,
        target_literal: Literal,
        predicates: list[Predicate],
        background_knowledge: dict[str, set[tuple]],
        allow_recursion: bool = False,
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
        self.allow_recursion = allow_recursion
        self.background_knowledge = background_knowledge
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

        uncovered_pos = pos_examples.copy()

        while uncovered_pos:
            # Learn a new clause to cover positive examples
            clause, best_pos_covered = self.new_clause(uncovered_pos, neg_examples)
            # Remove positive examples covered by this clause
            new_uncovered_pos = []
            for uncovered in uncovered_pos:
                for covered_posivive in best_pos_covered:
                    is_pos_example_covered = True
                    for key in uncovered.keys():
                        if uncovered[key] != covered_posivive[key]:
                            is_pos_example_covered = False
                            break
                    if is_pos_example_covered:
                        break

                if not is_pos_example_covered:
                    new_uncovered_pos.append(uncovered)

            uncovered_pos = new_uncovered_pos
            self.rules.append(clause)

            if self.allow_recursion:
                pass  # I think we need to add the clause to the background knowledge

    def new_clause(
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
        # Start with a clause with only the head (the target literal)
        clause: Clause = Clause(self.target_literal)

        # Combine positive and negative examples
        examples: list[tuple[bool, dict[str, Any]]] = [
            (False, ex) for ex in neg_examples
        ] + [(True, ex) for ex in uncovered_pos_examples]

        while [
            ex for is_pos, ex in examples if not is_pos
        ]:  # while negative examples are covered
            # Generate possible literals to add to the clause's body
            candidate_literals: list[Literal] = self.new_literals(clause)

            best_literal: Optional[Literal] = None
            best_gain: float = 0

            # Evaluate each candidate literal
            for literal in candidate_literals:
                if literal in clause.body:
                    continue  # Avoid cycles

                if literal == clause.head:
                    continue  # Avoid cycles

                extended_examples: list[tuple[bool, dict[str, Any]]] = []

                for example in examples:
                    extended_examples.extend(self.extend_example(example, literal))

                new_clause = copy.deepcopy(clause)
                new_clause.add_literal(literal)

                pos_covered = [ex for is_pos, ex in extended_examples if is_pos]
                neg_covered = [ex for is_pos, ex in extended_examples if not is_pos]

                if not pos_covered:
                    continue  # Must cover some positive examples

                gain = self.information_gain(
                    len(pos_covered),
                    len(neg_covered),
                    len(uncovered_pos_examples),
                    len(neg_examples),
                )

                if gain > best_gain:
                    best_gain = gain
                    best_literal = literal
                    best_pos_covered = pos_covered
                    best_neg_covered = neg_covered

            if best_literal is None:
                break  # No further improvement

            clause.add_literal(best_literal)

            examples = [(False, ex) for ex in best_neg_covered] + [
                (True, ex) for ex in best_pos_covered
            ]
            best_literal = None
            best_gain = 0

        return (clause, [ex for is_pos, ex in examples if is_pos])

    def extend_example(
        self, example: tuple[bool, dict[str, Any]], literal_to_add: Literal
    ) -> list[tuple[bool, dict[str, Any]]]:
        """
        Extend the example with the target predicate.
        """
        # example will be like (true, {V1: 1, V2: 2})
        # literal will be like Q(V1, V3)
        # need to loop over possible values
        # for V3 and add to examples if it satisfies the literal
        extended_examples: list[tuple[bool, dict[str, Any]]] = []
        new_vars: list[Variable] = []

        for arg in literal_to_add.args:
            # Check if the argument is a constant
            # if so append to args
            if isinstance(arg, Constant):
                continue

            # Check if the argument is a variable in the example
            # if so get value from example
            if isinstance(arg, Variable) and arg.name not in example[1]:
                new_vars.append(arg)
                continue

        if len(new_vars) == 0:
            if evaluate_literal(literal_to_add, example[1], self.background_knowledge):
                return [example]  # no new variables to add
            return []

        if len(new_vars) > 1:
            raise ValueError("We should only have at most one new variable to add")

        new_var: Variable = new_vars[0]

        for value in new_var.arg_type.possible_values:
            extended_example = (example[0], copy.deepcopy(example[1]))
            extended_example[1][new_var.name] = value
            if evaluate_literal(
                literal_to_add, extended_example[1], self.background_knowledge
            ):
                extended_examples.append(extended_example)

        return extended_examples

    def new_literals_for_predicate(
        self, predicate: Predicate, clause: Clause
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

        valid_vars_per_arg: list[list[Variable]] = [[] for _ in range(predicate.arity)]

        for var in current_vars:
            for i, arg_type in enumerate(predicate.arg_types):
                if arg_type == var.arg_type:
                    valid_vars_per_arg[i].append(var)

        # Introduce new variables (e.g., V1, V2)
        # We create a single additional variable for each type of argument
        new_vars_per_arg: list[list[Variable]] = [[] for _ in range(predicate.arity)]
        new_var_number = 1 + len(clause.variables)
        for i, arg_type in enumerate(predicate.arg_types):
            new_vars_per_arg[i].append(Variable(f"V{new_var_number}", arg_type))

        # Generate all combinations of existing variables
        var_combinations = [
            combo
            for combo in itertools.product(*valid_vars_per_arg)
            if len(set(combo)) == len(combo)  # avoid duplicates
        ]

        # append the combinations of the new variable with the other variables
        for i, new_vars in enumerate(new_vars_per_arg):
            new_vars_with_other_args_per_arg = (
                valid_vars_per_arg[:i] + [new_vars] + valid_vars_per_arg[i + 1 :]
            )
            var_combinations.extend(
                itertools.product(*new_vars_with_other_args_per_arg)
            )

        # TODO: We need to check types of variables to ensure they are
        # compatible with the predicate
        # TODO: How do we favour literals that are not adding a new variable?
        # Maybe this is done in the information gain calculation...

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

    def new_literals(self, clause: Clause) -> list[Literal]:
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
            literals.extend(self.new_literals_for_predicate(predicate, clause))

        return literals

        # TODO: extend to include inequality literals
        # TODO: see section on pruning in the paper

    def predict(self, example: dict[str, Any]) -> bool:
        """
        Predict if the example is positive based on learned rules.
        """
        for rule in self.rules:
            if rule.covers([example], self.background_knowledge):
                return True
        return False

    @staticmethod
    def information_gain(
        positive_new: int,
        negative_new: int,
        positive_old: int,
        negative_old: int,
    ) -> float:
        """
        Calculate the adjusted FOIL information gain when introducing new variables.
        :param p_old: Number of positive examples covered before adding the new literal.
        :param n_old: Number of negative examples covered before adding the new literal.
        :param p_new: Number of positive examples covered after adding the new literal.
        :param n_new: Number of negative examples covered after adding the new literal.
        :return: The adjusted information gain value.
        """
        # Avoid division by zero and log of zero
        if (
            positive_new == 0
            or (positive_new + negative_new) == 0
            or (positive_old + negative_old) == 0
        ):
            return 0

        # Compute probabilities
        prob_old = positive_old / (positive_old + negative_old)
        prob_new = positive_new / (positive_new + negative_new)

        # Ensure probabilities are positive
        if prob_new == 0 or prob_old == 0:
            return 0

        # Calculate the expansion factor f
        f = (
            positive_new / positive_old if positive_old > 0 else 1
        )  # Avoid division by zero

        # Ensure f is at least 1 to avoid negative gain due to floating point errors
        if f < 1:
            f = 1

        # Compute the adjusted information gain
        gain = positive_new * (
            math.log2(prob_new) - math.log2(prob_old)
        ) - positive_new * math.log2(f)

        return gain
