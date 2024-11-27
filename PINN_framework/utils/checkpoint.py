import orbax.checkpoint as ocp
from etils import epath
import os

def write_model(params, step, dir, init: bool = False):
    if init:
        if not dir.exists():
            dir.mkdir()
            # dir = ocp.test_utils.erase_and_create_empty(dir)
        return
    else:
        dir = epath.Path(dir)
    
    options = ocp.CheckpointManagerOptions(max_to_keep=1,
                                            create=True)
    mngr = ocp.CheckpointManager(directory=dir,
                                    options=options)
    saved = mngr.save(step=step,
                        args=ocp.args.StandardSave(params))
    mngr.wait_until_finished()
    return
    

def load_model(step_unused, dir):
    options = ocp.CheckpointManagerOptions(max_to_keep=1,
                                            create=True)
    mngr = ocp.CheckpointManager(directory=dir,
                                    options=options,
                                    item_handlers=ocp.StandardCheckpointHandler())
    step = max([eval(i) for i in os.listdir(dir)])
    return mngr.restore(step=step), step