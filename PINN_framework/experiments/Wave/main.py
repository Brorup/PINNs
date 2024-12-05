from typing import override
from time import perf_counter
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../../'))

import jax
import jax.numpy as jnp


from models.wave.pinn import WavePINN
from setup.parsers import parse_arguments
from utils.utils import timer

class PINN01(WavePINN):
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._set_loss(loss_term_fun_name="loss_terms")
        return

    def loss_terms(self,
                   params,
                   inputs: dict[str, jax.Array],
                   true_val: dict[str, jax.Array],
                   update_key: int | None = None,
                   **kwargs
                   ) -> jax.Array:
        """
        Retrieves all loss values and packs them together in a jax.Array.

        This function is ultimately called by an update function that is jitted and
        recompiled when the update_key argument changes. Therefore, one can write
        multiple different loss functions using if statements as below.
        """
        
        if update_key == 1:
            pass

        # Default update
        # Computes losses for domain and boundaries
        loss_coll = self.loss_coll(params, inputs["coll"], true_val=true_val.get("coll"))
        loss_bc = self.loss_bc(params, inputs["bc"], true_val=true_val.get("bc"))
        loss_ic0 = self.loss_ic0(params, inputs["ic0"], true_val=true_val.get("ic0"))
        loss_ic1 = self.loss_ic1(params, inputs["ic1"], true_val=true_val.get("ic1"))
        
        # Return 1D array of all loss values in the following order
        self.loss_names = ["coll"] + ["bc"] + ["ic0"] + ["ic1"]
        return jnp.array((loss_coll, loss_bc, loss_ic0, loss_ic1))

    def train(self, update_key = None, epochs: int | None = None, new_init: bool = False) -> None:
        """
        Method for training the model.

        This method initializes the optimzer and state,
        and then calls the update function for a number
        of times specified in the train_settings.
        """

        if new_init:
            del self.weights
            
        max_epochs = self.train_settings.iterations if epochs is None else epochs
        
        jitted_loss = jax.jit(self.loss_terms, static_argnames=self._static_loss_args)
        
        if not self.do_train:
            print("Model is not set to train")
            return
        else:
            if self._verbose.training:
                print("\nTraining:\n")
                print(f"Using the {self.train_settings.update_scheme} loss weighting scheme")
                print(f"Initial learning rate of {self.train_settings.learning_rate} with a decay of {self.train_settings.decay_rate} every {self.train_settings.decay_steps} steps")
                print(f"Training for {max_epochs} epochs\n\n")
                sys.stdout.flush()


        # Start time
        t0 = perf_counter()
        for epoch in range(max_epochs):
            
            self.get_weights(epoch, 
                             jitted_loss, 
                             self.params, 
                             self.train_points, 
                             true_val=self.train_true_val, 
                             update_key=update_key)
            
            for batch_num, (xy_batch, u_batch) in enumerate(iter(self.full_batch_dataset)):
                
                self.train_points_batch["coll"] = xy_batch
                self.train_true_val_batch["coll"] = u_batch
                
                # Update step
                self.params, self.opt_state, total_loss, loss_terms = self.update(opt_state=self.opt_state,
                                                                                                params=self.params,
                                                                                                inputs=self.train_points_batch,
                                                                                                weights=self.weights,
                                                                                                true_val=self.train_true_val_batch,
                                                                                                update_key=update_key,
                                                                                                start_time=t0,
                                                                                                epoch=epoch,
                                                                                                learning_rate=self.schedule(epoch),
                                                                                                batch_num=batch_num
                                                                                                )
            
            self.do_every(epoch, jitted_loss, 
                             params=self.params, 
                             inputs=self.train_points, 
                             true_val=self.train_true_val, 
                             update_key=update_key)

        if self._verbose.training:
            print("###############################################################\n\n")
        return


if __name__ == "__main__":
    
    raw_settings = parse_arguments()
    pinn = PINN01(raw_settings)
    pinn.sample_points()
    pinn.train()
    pinn.eval()
    pinn.plot_results()
