from amlgym.algorithms.rosame import Rosame_Runner


class PORosame_Runner(Rosame_Runner):

    def prepare_rosame_data(self,observation):
        """
        Prepares structured data from traces to be used within the ROSAME framework.
        Handles partial observability by encoding predicates as:
        - 1.0 if predicate is positive and unmasked
        - 0.5 if predicate is masked (uncertain)
        - 0.0 if predicate is negative and unmasked

        returns:
        - encoded pre_state
        - encoded next_state
        - encoded action
        """
        steps_action = []
        steps_state1, steps_state2 = [], []

        for component in observation.components:
            action_lst = component.grounded_action_call.__str__()[1:-1].split()
            if len(action_lst) != len(set(action_lst)):
                continue

            steps_action.append(self.check_action(component.grounded_action_call.__str__()[1:-1]))

            # Process previous state
            pre_positive_preds = []  # Positive predicates that are unmasked
            pre_masked_preds = []    # Predicates that are masked (pos or neg)
            pre_negative_preds = []  # Negative predicates that are unmasked

            for _, val in component.previous_state.state_predicates.items():
                for pred in val:
                    pred_str = self.check_predicate(pred.untyped_representation[1:-1])

                    if pred.is_masked:
                        pre_masked_preds.append(pred_str)
                    elif pred.is_positive:
                        pre_positive_preds.append(pred_str)
                    else:  # is_negative and not masked
                        pre_negative_preds.append(pred_str)

            # Process next state
            next_positive_preds = []  # Positive predicates that are unmasked
            next_masked_preds = []    # Predicates that are masked (pos or neg)
            next_negative_preds = []  # Negative predicates that are unmasked

            for _, val in component.next_state.state_predicates.items():
                for pred in val:
                    pred_str = self.check_predicate(pred.untyped_representation[1:-1])

                    if pred.is_masked:
                        next_masked_preds.append(pred_str)
                    elif pred.is_positive:
                        next_positive_preds.append(pred_str)
                    else:  # is_negative and not masked
                        next_negative_preds.append(pred_str)

            # Encode states with partial observability
            # 1.0 if in positive_preds, 0.5 if in masked_preds, 0.0 if in negative_preds
            state1 = []
            for p in self.rosame.propositions:
                if p in pre_positive_preds:
                    state1.append(1.0)
                elif p in pre_masked_preds:
                    state1.append(0.5)
                else:  # Either in negative_preds or not observed at all
                    state1.append(0.0)

            state2 = []
            for p in self.rosame.propositions:
                if p in next_positive_preds:
                    state2.append(1.0)
                elif p in next_masked_preds:
                    state2.append(0.5)
                else:  # Either in negative_preds or not observed at all
                    state2.append(0.0)

            steps_state1.append(state1)
            steps_state2.append(state2)

        return steps_state1, steps_action, steps_state2
