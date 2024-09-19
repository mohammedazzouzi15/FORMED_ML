import GNN_models.main_fair_chem as main_fair_chem
import argparse
from fairchem.core.common.utils import (
    new_trainer_context,
)
from fairchem.core.tasks.task import BaseTask
import optuna
from fairchem.core.common.registry import registry
from optuna.trial import TrialState
from OptunaTrained import OptunaTrained


@registry.register_task("validate")
class OptunaTasks(BaseTask):
    def run(self,trial) -> None:
        try:
            self.trainer.train(trial,
                disable_eval_tqdm=self.config.get("hide_eval_progressbar", False)
            )
        except RuntimeError as e:
            self._process_error(e)
            raise e
def model_hyperparameters(trial,model):
    model.hidden_channels = trial.suggest_categorical("hidden_channels", [ 64, 128, 256, 512, 1024])
    model.num_interactions = trial.suggest_int("num_interactions", 4, 12)
    model.cutoff = trial.suggest_int("cutoff", 2, 8)

def main():
    study_name = "schnet_Es1_3"
    def evaluation_function(trial):
        main_fair_chem.setup_logging()
        parser: argparse.ArgumentParser = main_fair_chem.flags.get_parser()
        args = parser.parse_args(["--mode", "validate", "--config-yml", "schnet.yml"])

        #args, override_args = parser.parse_known_args()

        config = main_fair_chem.build_config(args,{})
        with new_trainer_context(config=config) as ctx:
            config = ctx.config
            
            task = ctx.task
            trainer = ctx.trainer
        task.setup(trainer)
        model_hyperparameters(trial, task.trainer.model)
        task.run(trial)
        return task.trainer.best_val_metric
    print("running")
    study = optuna.create_study(direction="minimize",storage="sqlite:///optuna.db", study_name=study_name,load_if_exists=True)
    study.optimize(evaluation_function, n_trials=20, timeout=6000)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

if __name__ == "__main__":
    main()
