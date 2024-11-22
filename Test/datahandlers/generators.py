from collections.abc import Sequence
import math

import numpy as np
from scipy.stats.qmc import Sobol
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import flax
import torch.utils.data

from datahandlers.samplers import sample_line
from utils.utils import limits2vertices, remove_points, keep_points


def generate_interval_points(key: jax.random.PRNGKey,
                             xlim: Sequence[float],
                             num_points: int,
                             sobol: bool = True,
                             round_up: bool = False):
    
    s_key, key = jax.random.split(key, 2)
    if sobol:
        # Use Sobol QMC sampling (convert key to seed, to make sampling "deterministic")
        xp = Sobol(1, seed=int(jax.random.randint(s_key, (), 0,
                                                  jnp.iinfo(jnp.int32).max))
            ).random_base2(math.ceil(jnp.log2(num_points)))
        
        if not round_up:
            xp = xp[:num_points]

        return jnp.array(xp*(xlim[1]-xlim[0]) + xlim[0])
    
    # Use uniform sampling
    return jax.random.uniform(s_key, (num_points, 1), minval=xlim[0], maxval=xlim[1])
    



def generate_rectangle_points(key: jax.random.PRNGKey,
                              xlim: Sequence[float],
                              ylim: Sequence[float],
                              num_points: int | Sequence[int],
                              domain_type: str = "full",
                              radius: float = 2.0
                              ) -> tuple:
    """
    Order of rectangle sides: Lower horizontal, right vertical, upper horizontal, left vertical.
    """
    if isinstance(num_points, int):
        N = [num_points] * 4
    elif isinstance(num_points, Sequence):
        if len(num_points) == 1:
            N = num_points * 4
        elif len(num_points) == 4:
            N = num_points
        else:
            raise ValueError(f"Wrong length of 'num_points': f{len(num_points)}. Sequence length must be either 1 or 4.")
    else:
        raise ValueError("Argument 'num_points' must be int or tuple.")
    
    key, *keys = jax.random.split(key, 6) if "half" in domain_type.lower() else jax.random.split(key, 5)

    c = lambda p1, p2: jnp.concatenate((p1, p2), axis=0)

    match domain_type.lower():
        case "full":
            v = [
                ([xlim[0], ylim[0]], [xlim[1], ylim[0]]), # Lower horizontal
                ([xlim[1], ylim[0]], [xlim[1], ylim[1]]), # Right vertical
                ([xlim[1], ylim[1]], [xlim[0], ylim[1]]), # Upper horizontal
                ([xlim[0], ylim[1]], [xlim[0], ylim[0]])  # Left vertical
            ]
            return tuple(sample_line(keys[i], v[i], shape=(N[i], 1)) for i in range(4))
        
        case "half-upper":
            v = [
                ([xlim[0], ylim[0]], [-radius, ylim[0]]), # Lower horizontal
                ([ radius, ylim[0]], [xlim[1], ylim[0]]), # Lower horizontal
                ([xlim[1], ylim[0]], [xlim[1], ylim[1]]), # Right vertical
                ([xlim[1], ylim[1]], [xlim[0], ylim[1]]), # Upper horizontal
                ([xlim[0], ylim[1]], [xlim[0], ylim[0]])  # Left vertical
            ]
            N = [math.ceil(N[0] / 2), math.floor(N[0] / 2), *N[1:]]
            points = [sample_line(keys[i], v[i], shape=(N[i], 1)) for i in range(5)]
            return (c(points[0], points[1]), points[2], points[3], points[4])

        case "half-lower":
            v = [
                ([xlim[0], ylim[0]], [xlim[1], ylim[0]]), # Lower horizontal
                ([xlim[1], ylim[0]], [xlim[1], ylim[1]]), # Right vertical
                ([xlim[1], ylim[1]], [ radius, ylim[1]]), # Upper horizontal
                ([-radius, ylim[1]], [xlim[0], ylim[1]]), # Upper horizontal
                ([xlim[0], ylim[1]], [xlim[0], ylim[0]])  # Left vertical
            ]
            N = [*N[:2], math.ceil(N[2] / 2), math.floor(N[2] / 2), N[3]]
            points = [sample_line(keys[i], v[i], shape=(N[i], 1)) for i in range(5)]
            return (points[0], points[1], c(points[2], points[3]), points[4])
        
        case "half-left":
            v = [
                ([xlim[0], ylim[0]], [xlim[1], ylim[0]]), # Lower horizontal
                ([xlim[1], ylim[0]], [xlim[1], -radius]), # Right vertical
                ([xlim[1],  radius], [xlim[1], ylim[1]]), # Right vertical
                ([xlim[1], ylim[1]], [xlim[0], ylim[1]]), # Upper horizontal
                ([xlim[0], ylim[1]], [xlim[0], ylim[0]])  # Left vertical
            ]
            N = [N[0], math.ceil(N[1] / 2), math.floor(N[1] / 2), *N[2:]]
            points = [sample_line(keys[i], v[i], shape=(N[i], 1)) for i in range(5)]
            return (points[0], c(points[1], points[2]), points[3], points[4])
        
        case "half-right":
            v = [
                ([xlim[0], ylim[0]], [xlim[1], ylim[0]]), # Lower horizontal
                ([xlim[1], ylim[0]], [xlim[1], ylim[1]]), # Right vertical
                ([xlim[1], ylim[1]], [xlim[0], ylim[1]]), # Upper horizontal
                ([xlim[0], ylim[1]], [xlim[0],  radius]), # Left vertical
                ([xlim[0], -radius], [xlim[0], ylim[0]])  # Left vertical
            ]
            N = [*N[:3], math.ceil(N[0] / 2), math.floor(N[0] / 2)]
            points = [sample_line(keys[i], v[i], shape=(N[i], 1)) for i in range(5)]
            return (points[0], points[1], points[2], c(points[3], points[4]))
        
        case "quarter-1":
            v = [
                ([ radius,       0], [xlim[1],       0]), # Lower horizontal
                ([xlim[1],       0], [xlim[1], ylim[1]]), # Right vertical
                ([xlim[1], ylim[1]], [      0, ylim[1]]), # Upper horizontal
                ([      0, ylim[1]], [      0,  radius])  # Left vertical
            ]
            return tuple(sample_line(keys[i], v[i], shape=(N[i], 1)) for i in range(4))
        
        case "quarter-2":
            v = [
                ([xlim[0],       0], [-radius,       0]), # Lower horizontal
                ([      0,  radius], [      0, ylim[1]]), # Right vertical
                ([      0, ylim[1]], [xlim[0], ylim[1]]), # Upper horizontal
                ([xlim[0], ylim[1]], [xlim[0],       0])  # Left vertical
            ]
            return tuple(sample_line(keys[i], v[i], shape=(N[i], 1)) for i in range(4))
        
        case "quarter-3":
            v = [
                ([xlim[0], ylim[0]], [      0, ylim[0]]), # Lower horizontal
                ([      0, ylim[0]], [      0, -radius]), # Right vertical
                ([-radius,       0], [xlim[0],       0]), # Upper horizontal
                ([xlim[0],       0], [xlim[0], ylim[0]])  # Left vertical
            ]
            return tuple(sample_line(keys[i], v[i], shape=(N[i], 1)) for i in range(4))
        
        case "quarter-4":
            v = [
                ([      0, ylim[0]], [xlim[1], ylim[0]]), # Lower horizontal
                ([xlim[1], ylim[0]], [xlim[1],       0]), # Right vertical
                ([xlim[1],       0], [ radius,       0]), # Upper horizontal
                ([      0, -radius], [      0, ylim[0]])  # Left vertical
            ]
            return tuple(sample_line(keys[i], v[i], shape=(N[i], 1)) for i in range(4))
        
        case _:
            raise ValueError(f"Unknown domain type: '{domain_type}'.")


def generate_circle_points(key: jax.random.PRNGKey,
                           radius: float,
                           num_points: int,
                           angle_interval: Sequence[float] | None = None,
                           sobol: bool = True
                           ) -> jax.Array:
    theta_min = 0
    theta_max = 2*jnp.pi
    if angle_interval is not None:
        theta_min = angle_interval[0]
        theta_max = angle_interval[1]
    
    if sobol:
        theta = sample_line(key, (theta_min, theta_max),shape=(num_points, 1))
    else:
        theta = jax.random.uniform(key, (num_points, 1), minval=theta_min, maxval=theta_max)
    
    xc = radius*jnp.cos(theta)
    yc = radius*jnp.sin(theta)
    xyc = jnp.stack([xc, yc], axis=1).reshape((-1,2))
    return xyc


def generate_collocation_points(key: jax.random.PRNGKey,
                                xlim: Sequence[float],
                                ylim: Sequence[float],
                                num_coll: int,
                                sobol: bool = True,
                                round_up: bool = True) -> jax.Array:
    if num_coll <= 0:
        return jnp.empty((0, 2))
    
    if sobol:
        # Use Sobol QMC sampling (convert key to seed, to make sampling "deterministic")
        s_key, key = jax.random.split(key, 2)
        xp = Sobol(2, seed=int(jax.random.randint(s_key, (), 0,
                                                  jnp.iinfo(jnp.int32).max))
            ).random_base2(math.ceil(jnp.log2(num_coll)))
        
        xp[:, 0] = xp[:, 0]*(xlim[1]-xlim[0]) + xlim[0]
        xp[:, 1] = xp[:, 1]*(ylim[1]-ylim[0]) + ylim[0]

        if not round_up:
            xp = xp[:num_coll]

        return jnp.array(xp)
    
    # Uniform sampling
    shape_pde = (num_coll, 1)
    x_key, y_key = jax.random.split(key, 2)
    x_train = jax.random.uniform(x_key, shape_pde, minval=xlim[0], maxval=xlim[1])
    y_train = jax.random.uniform(y_key, shape_pde, minval=ylim[0], maxval=ylim[1])
    xp = jnp.stack([x_train, y_train], axis=1).reshape((-1,2))
    
    return xp
    

def generate_extra_points(keyr, keytheta, radius, num_extra,
                          angle_interval: Sequence[float] | None = None,
                          intensity: float = 10.0):
    if num_extra <= 0:
        return jnp.empty((0, 2))
    
    theta_min = 0
    theta_max = 2*jnp.pi
    if angle_interval is not None:
        theta_min = angle_interval[0]
        theta_max = angle_interval[1]
    
    theta_rand = jax.random.uniform(keytheta, (num_extra, 1), minval=theta_min, maxval=theta_max)
    r_rand = jax.random.chisquare(keyr, df=2, shape=(num_extra, 1)) / intensity + radius
    xy_extra = jnp.stack([r_rand*jnp.cos(theta_rand), r_rand*jnp.sin(theta_rand)], axis=1).reshape((-1,2))
    return xy_extra


def generate_collocation_points_with_hole(key: jax.random.PRNGKey,
                                          radius: float, 
                                          xlim_all: Sequence[float],
                                          ylim_all: Sequence[float],
                                          points: int | Sequence[int] | None,
                                          sobol: bool = True,
                                          round_up = True,
                                          domain_type: str = "full"
                                          ) -> jax.Array | tuple[jax.Array]:
    """
    This function samples points in the inner of the domain.

    
    Domain types:
    'full':         The whole domain ([xlim] X [ylim]).
    'half-upper':   The upper half of the domain.
                    Replace 'upper' with 'lower' / 'left' / 'right' to get other halfs.
    'quarter-1':    First quadrant (x > 0, y > 0) of domain.
                    Replace '1' with '2' / '3' / '4' to get other quadrants.
    """
    if points is None:
        return jnp.empty((0,))
    
    if not isinstance(points, Sequence):
        points = [points]

    match domain_type.lower():
        case "full":
            xlim = xlim_all
            ylim = ylim_all
            angle_interval = [0, 2*jnp.pi]
        case "half-upper":
            xlim = xlim_all
            ylim = [0, ylim_all[1]]
            angle_interval = [0, jnp.pi]
        case "half-lower":
            xlim = xlim_all
            ylim = [ylim_all[0], 0]
            angle_interval = [jnp.pi, 2*jnp.pi]
        case "half-left":
            xlim = [xlim_all[0], 0]
            ylim = ylim_all
            angle_interval = [0.5*jnp.pi, 1.5*jnp.pi]
        case "half-right":
            xlim = [0, xlim_all[1]]
            ylim = ylim_all
            angle_interval = [-0.5*jnp.pi, 0.5*jnp.pi]
        case "quarter-1":
            xlim = [0, xlim_all[1]]
            ylim = [0, ylim_all[1]]
            angle_interval = [0, 0.5*jnp.pi]
        case "quarter-2":
            xlim = [xlim_all[0], 0]
            ylim = [0, ylim_all[1]]
            angle_interval = [0.5*jnp.pi, jnp.pi]
        case "quarter-3":
            xlim = [xlim_all[0], 0]
            ylim = [ylim_all[0], 0]
            angle_interval = [jnp.pi, 1.5*jnp.pi]
        case "quarter-4":
            xlim = [0, xlim_all[1]]
            ylim = [ylim_all[0], 0]
            angle_interval = [1.5*jnp.pi, 2*jnp.pi]
        case _:
            raise ValueError(f"Unknown domain type: '{domain_type}'.")
        

    num_coll = points[0]

    # Initial coll point gen
    key, key_coll = jax.random.split(key)
    xy_coll = generate_collocation_points(key_coll, xlim, ylim, num_coll, sobol=sobol, round_up=round_up)
    xy_coll = remove_points(xy_coll, lambda p: jnp.linalg.norm(p, axis=-1) <= radius)
    
    # Filler coll point gen
    pnum = xy_coll.shape[0]
    while pnum < num_coll:
        key, key_coll = jax.random.split(key_coll)
        tmp = generate_collocation_points(key_coll, xlim, ylim, num_coll, sobol=sobol, round_up=True)
        tmp = remove_points(tmp, lambda p: jnp.linalg.norm(p, axis=-1) <= radius)
        xy_coll = jnp.concatenate((xy_coll, tmp))
        pnum = xy_coll.shape[0]
    xy_coll = xy_coll[:num_coll]

    # Return if no extra points should be generated
    if len(points) == 1:
        return xy_coll
    
    num_extra = points[1]

    # Initial extra point gen
    key, keytheta, keyr = jax.random.split(key, 3)
    xy_extra = generate_extra_points(keyr, keytheta, radius, num_extra, angle_interval=angle_interval)
    xy_extra = keep_points(xy_extra, lambda p: jnp.logical_and(jnp.logical_and(p[:, 0] >= xlim[0], p[:, 0] <= xlim[1]),
                                                               jnp.logical_and(p[:, 1] >= ylim[0], p[:, 1] <= ylim[1])))
    
    # Filler extra point gen
    pnum = xy_extra.shape[0]
    while pnum < num_extra:
        key, keytheta, keyr = jax.random.split(key, 3)
        tmp = generate_extra_points(keyr, keytheta, radius, num_extra, angle_interval=angle_interval)
        tmp = keep_points(xy_extra, lambda p: jnp.logical_and(jnp.logical_and(p[:, 0] >= xlim[0], p[:, 0] <= xlim[1]),
                                                              jnp.logical_and(p[:, 1] >= ylim[0], p[:, 1] <= ylim[1])))
        xy_extra = jnp.concatenate((xy_extra, tmp))
        pnum = xy_extra.shape[0]
    xy_extra = xy_extra[:num_extra]
    
    # Collect all points
    return jnp.concatenate((xy_coll, xy_extra))
    

def generate_rectangle_with_hole(key: jax.random.PRNGKey,
                                 radius: float, 
                                 xlim: Sequence[float],
                                 ylim: Sequence[float],
                                 num_coll: int | Sequence[int],
                                 num_rect: int | Sequence[int],
                                 num_circ: int,
                                 sobol: bool = True,
                                 round_up = True,
                                 domain_type: str = "full"
                                 ) -> dict[str, jax.Array | tuple[jax.Array]]:
    """
    Main function for generating necessary sample points for the plate-with-hole problem.

    The function generates 
    """
    angle_intervals = {
        "full":       [        0.0, 2.0*jnp.pi],
        "half-upper": [        0.0,     jnp.pi],
        "half-lower": [     jnp.pi, 2.0*jnp.pi],
        "half-left":  [ 0.5*jnp.pi, 1.5*jnp.pi],
        "half-right": [-0.5*jnp.pi, 0.5*jnp.pi],
        "quarter-1":  [        0.0, 0.5*jnp.pi],
        "quarter-2":  [ 0.5*jnp.pi,     jnp.pi],
        "quarter-3":  [     jnp.pi, 1.5*jnp.pi],
        "quarter-4":  [ 1.5*jnp.pi, 2.0*jnp.pi],
    }
    
    try:
        angle_interval = angle_intervals[domain_type]
    except KeyError as k:
        raise ValueError(f"Unknown domain type: '{domain_type}'.")

    key, rectkey, circkey, collkey, permkey = jax.random.split(key, 5)

    xy_coll = generate_collocation_points_with_hole(collkey, radius, xlim, ylim, num_coll, sobol=sobol, domain_type=domain_type, round_up=round_up)
    xy_rect = generate_rectangle_points(rectkey, xlim, ylim, num_rect, domain_type=domain_type, radius=radius)
    xy_circ = generate_circle_points(circkey, radius, num_circ, angle_interval=angle_interval)
    # xy_test = generate_collocation_points_with_hole(testkey, radius, xlim, ylim, num_test)
    return {"coll": xy_coll, "rect": xy_rect, "circ": xy_circ}
    

def generate_rectangle(key: jax.random.PRNGKey,
                       xlim: Sequence[float],
                       ylim: Sequence[float],
                       num_coll: int | Sequence[int],
                       num_rect: int | Sequence[int],
                       sobol: bool = True) -> dict[str, jax.Array | tuple[jax.Array]]:
    """
    Main function for generating necessary sample points for the square problem.
    """
    
    if isinstance(num_coll, list):
        num_coll = num_coll[0]

    key, rectkey, collkey, permkey = jax.random.split(key, 4)

    xy_coll = generate_collocation_points(collkey, xlim, ylim, num_coll, sobol=sobol)
    xy_coll = jax.random.permutation(permkey, xy_coll)
    xy_rect = generate_rectangle_points(rectkey, xlim, ylim, num_rect)
    return {"coll": xy_coll, "rect": xy_rect}


def resample(new_arr: jax.Array, new_loss: jax.Array, num_keep: int):
    """
    Utility function for choosing the points with
    highest loss.

    input:
        new_arr:
            The array to choose points from.
        
        new_loss:
            The losses to base the choice on.
        
        num_keep:
            The number of sampled points to keep.
        
    """

    num_keep = min(num_keep, new_loss.ravel().shape[0])
    idx = jnp.argpartition(new_loss.ravel(), kth=-num_keep)
    return new_arr[idx[-num_keep:]]


def resample_idx(new_arr: jax.Array, new_loss: jax.Array, num_throwaway: int):
    """
    Utility function for finding the indices of the points with the lowest loss

    input:
        new_arr:
            The array to choose points from.
        
        new_loss:
            The losses to base the choice on.
        
        num_throwaway:
            The number of sampled points to not keep.
        
    """

    num_throwaway = min(num_throwaway, new_loss.ravel().shape[0])
    idx = jnp.argpartition(new_loss.ravel(), kth=-num_throwaway)
    return idx[:num_throwaway]


class JaxDataset:
    def __init__(self, key, xy, u, batch_size):
        self.key, key = jax.random.split(key, 2)
        self.xy = jax.random.permutation(key, xy, axis=0)
        if u is not None:
            self.u = jax.random.permutation(key, u, axis=0)
        else:
            self.u = None
        self.count = 0
        if batch_size is None:
            self.batch_size = self.xy.shape[0]
        else:
            self.batch_size = batch_size
        return

    def _permute(self):
        self.key, key = jax.random.split(self.key, 2)
        self.xy = jax.random.permutation(key, self.xy, axis=0)
        if self.u is not None:
            self.u = jax.random.permutation(key, self.u, axis=0)
        self.count = 0
        return

    def __len__(self):
        return self.xy.shape[0]
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.count+self.batch_size  > len(self):
            self._permute()
            raise StopIteration
        
        if self.u is not None:
            batch = tuple(arr[self.count:self.count+self.batch_size] for arr in [self.xy, self.u])
        else:
            batch = tuple([self.xy[self.count:self.count+self.batch_size], None])
        self.count += self.batch_size
        return batch
















def numpy_collate(batch):
    return jtu.tree_map(np.asarray, torch.utils.data.default_collate(batch))


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_dict: dict, seed: int = 0) -> None:
        # Set data and "labels"
        # Input structure: One pair for each condition
        # data  =  {
        #           "pde": (jnp.ndarray <- collocation points, jnp.ndarray <- RHS if any),
        #           "xy": (jnp.ndarray <- [x, y, z, ...], jnp.ndarray <- [u_xy(x, y, z, ...), v_xy(x, ...), ...]),
        #           "xx": (jnp.ndarray <-   ---||---    , jnp.ndarray <-   ---||---                       ),
        #           .
        #           .
        #           .
        #           "val", (jnp.ndarray <- x_val, jnp.ndarray <- u_val)
        #           }
        #

        self.key = jax.random.PRNGKey(seed)

        # Data types 
        self.data_keys = [key for key in data_dict.keys()]
        
        # Extract data lengths and calculate 'global' indices of data
        length = [value[0].shape[0] for value in data_dict.values()]
        indx = [jnp.arange(sum(length[:i]), sum(length[:i+1])) for i, _ in enumerate(length)]
        self.data_length_accum = [sum(length[:i+1]) for i, _ in enumerate(length)]

        # Create dictionaries with data lengths and indices
        self.data_length = flax.core.FrozenDict([(data_key, data_length) for data_key, data_length in zip(self.data_keys, length)])
        self.data_indx = flax.core.FrozenDict([(data_key, data_indx) for data_key, data_indx in zip(self.data_keys, indx)])
        self.indx_dict = flax.core.FrozenDict([(key, jnp.arange(value[0].shape[0])) for i, (key, value) in enumerate(data_dict.items())])

        # Flattened/concatenated dataset
        self.x = np.vstack([data[0] for data in data_dict.values()])
        self.u = np.vstack([data[1] for data in data_dict.values()])
        
        return

    
    def __len__(self):
        # Define "length" of dataset
        return self.x.shape[0]

    def __getitem__(self, index):
        # Define which data to fetch based on input index
        type_idx = np.searchsorted(self.data_length_accum, index)
        # batch = (self.x[index], self.u[index], type_idx)
        return index, self.x[index], self.u[index], type_idx
        # return index, index

    def __repr__(self):
        num_points = 5
        r = 4
        repr_str = "\n" + "--------" * r + "  DATASET  " + "--------" * r + "\n\n"
        repr_str += f"Printing first {num_points} data points for each entry:\n\n"
        for key, idx in self.data_indx.items():
            n = min(len(idx), num_points)
            repr_str += (16*r+11) * "-"
            repr_str += f"\nType: '{key}'"
            repr_str += "\n\nData points:\n"
            repr_str += str(self.x[idx[:n]])
            repr_str += "\n\nFunction values\n"
            repr_str += str(self.u[idx[:n]])
            repr_str += "\n\n\n"
        repr_str += "\n"
        repr_str += (16*r+11) * "-"
        repr_str += "\n"
        return repr_str


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1,
                 shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
              batch_size=batch_size,
              shuffle=shuffle,
              sampler=sampler,
              batch_sampler=batch_sampler,
              num_workers=num_workers,
              collate_fn=numpy_collate,
              pin_memory=pin_memory,
              drop_last=drop_last,
              timeout=timeout,
              worker_init_fn=worker_init_fn)


if __name__ == "__main__":
    from utils.utils import out_shape

    key = jax.random.PRNGKey(123)
    points = generate_rectangle_points(key, [0, 1], [0, 1], 150)
    test_fun = lambda p: (p[:, 0]**2 + p[:, 1]**2).reshape(-1, 1)
    data_dict = {"yy": (points[0], test_fun(points[0])),
                 "xy": (points[1], test_fun(points[1])),
                 "xx": (points[0], test_fun(points[0])),}
    dataset = Dataset(data_dict)
    batch_size = 16
    print(dataset)

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    for i, (global_indices, x, u, tidx) in enumerate(dataloader):
        print("Batch", i, "with indices", global_indices, "\n", x, "\n", u, "\n", tidx, "\n\n")
    