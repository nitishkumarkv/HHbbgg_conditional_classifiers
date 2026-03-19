import argparse
import unittest

from run_multiclass_strategy import validate_requested_actions


class RunMulticlassStrategyTests(unittest.TestCase):
    def test_no_action_selected_raises(self):
        parser = argparse.ArgumentParser()
        args = argparse.Namespace(
            prep_inputs_for_training=False,
            prepare_inputs_pred_sim=False,
            prepare_inputs_pred_data=False,
            prepare_inputs_pred_sys=False,
            submit_prepare_to_condor=False,
            perform_training=False,
            train_best_model=False,
            plot_training_results=False,
            get_permutation_importance=False,
            get_predictions=False,
            get_predictions_sys=False,
            test_mass_sculpting=False,
            get_data_mc_plots=False,
            get_score_shape_diff_kl=False,
            perform_categorisation=False,
            submit_training_to_condor=False,
            condor_better_analyze=None,
        )
        with self.assertRaises(SystemExit):
            validate_requested_actions(args, parser)

    def test_submit_training_to_condor_requires_training_action(self):
        parser = argparse.ArgumentParser()
        args = argparse.Namespace(
            prep_inputs_for_training=False,
            prepare_inputs_pred_sim=False,
            prepare_inputs_pred_data=False,
            prepare_inputs_pred_sys=False,
            submit_training_to_condor=True,
            submit_prepare_to_condor=False,
            train_best_model=False,
            perform_training=False,
            plot_training_results=False,
            get_permutation_importance=False,
            get_predictions=False,
            get_predictions_sys=False,
            test_mass_sculpting=False,
            get_data_mc_plots=False,
            get_score_shape_diff_kl=False,
            perform_categorisation=False,
            condor_better_analyze=None,
        )
        with self.assertRaises(SystemExit):
            validate_requested_actions(args, parser)

    def test_submit_training_to_condor_allows_train_best_model(self):
        parser = argparse.ArgumentParser()
        args = argparse.Namespace(
            prep_inputs_for_training=False,
            prepare_inputs_pred_sim=False,
            prepare_inputs_pred_data=False,
            prepare_inputs_pred_sys=False,
            submit_training_to_condor=True,
            submit_prepare_to_condor=False,
            train_best_model=True,
            perform_training=False,
            plot_training_results=False,
            get_permutation_importance=False,
            get_predictions=False,
            get_predictions_sys=False,
            test_mass_sculpting=False,
            get_data_mc_plots=False,
            get_score_shape_diff_kl=False,
            perform_categorisation=False,
            condor_better_analyze=None,
        )
        validate_requested_actions(args, parser)

    def test_submit_prepare_to_condor_requires_prepare_action(self):
        parser = argparse.ArgumentParser()
        args = argparse.Namespace(
            prep_inputs_for_training=False,
            prepare_inputs_pred_sim=False,
            prepare_inputs_pred_data=False,
            prepare_inputs_pred_sys=False,
            submit_prepare_to_condor=True,
            perform_training=False,
            submit_training_to_condor=False,
            train_best_model=False,
            plot_training_results=False,
            get_permutation_importance=False,
            get_predictions=False,
            get_predictions_sys=False,
            test_mass_sculpting=False,
            get_data_mc_plots=False,
            get_score_shape_diff_kl=False,
            perform_categorisation=False,
            condor_better_analyze=None,
        )
        with self.assertRaises(SystemExit):
            validate_requested_actions(args, parser)

    def test_submit_prepare_to_condor_with_prepare_inputs_is_allowed(self):
        parser = argparse.ArgumentParser()
        args = argparse.Namespace(
            prep_inputs_for_training=True,
            prepare_inputs_pred_sim=False,
            prepare_inputs_pred_data=False,
            prepare_inputs_pred_sys=False,
            submit_prepare_to_condor=True,
            perform_training=False,
            submit_training_to_condor=False,
            train_best_model=False,
            plot_training_results=False,
            get_permutation_importance=False,
            get_predictions=False,
            get_predictions_sys=False,
            test_mass_sculpting=False,
            get_data_mc_plots=False,
            get_score_shape_diff_kl=False,
            perform_categorisation=False,
            condor_better_analyze=None,
        )
        validate_requested_actions(args, parser)


if __name__ == "__main__":
    unittest.main()
