# Heavily modified from haiku/examples/imagenet/train.py

import preprocess
import gpu
import os
import pickle
import contextlib
import functools
import timeit
from typing import Iterable, Mapping, NamedTuple, Tuple
import tensorflow_datasets as tfds
from absl import app
from absl import flags
from absl import logging
import haiku as hk
import jmp
import numpy as np
import optax
import tree
from preprocess import *
import resnet as rn

# Hyper parameters.
flags.DEFINE_integer('img_size_num', 7, help='') # see img_sizes in __main__. 256 went to 2.91 # 11 for 350
flags.DEFINE_integer('batch_size', 512, help='') #For 128 img size, resnet50, this fits in VRAM (YMMV)
flags.DEFINE_integer('run_num', 0, help='')
flags.DEFINE_float('model_bn_decay', 0.9, help='')
flags.DEFINE_float('train_weight_decay', 4e-5, help='')
flags.DEFINE_float('dropout_rate', 0.25, help='') #0.25 seems to work well
flags.DEFINE_integer('layer_size', 256, help='')
flags.DEFINE_bool('model_resnet_v2', True, help='')
flags.DEFINE_integer('resnet_num', 5, help='') # see resnets list in __main__
flags.DEFINE_float('optimizer_momentum', 0.9, help='')
flags.DEFINE_bool('optimizer_use_nesterov', True, help='')
flags.DEFINE_integer('train_eval_every', 500, help='')
flags.DEFINE_integer('train_init_random_seed', 42, help='')
flags.DEFINE_integer('train_log_every', 100, help='')
flags.DEFINE_integer('train_epochs', 1000, help='')
flags.DEFINE_integer('train_lr_warmup_epochs', 5, help='')
flags.DEFINE_float('train_lr_init', 1e-4, help='')
flags.DEFINE_float('train_smoothing', .1, lower_bound=0, upper_bound=1, help='')
flags.DEFINE_string('mp_policy', 'p=f32,c=f32,o=f32', help='') # change to c=f16 for half precision
flags.DEFINE_string('mp_bn_policy', 'p=f32,c=f32,o=f32', help='')
flags.DEFINE_enum('mp_scale_type', 'NoOp', ['NoOp', 'Static', 'Dynamic'],
                  help='')
flags.DEFINE_float('mp_scale_value', 2 ** 15, help='')
flags.DEFINE_bool('mp_skip_nonfinite', False, help='')
flags.DEFINE_bool('dataset_transpose', False, help='')
flags.DEFINE_bool('dataset_zeros', False, help='')
FLAGS = flags.FLAGS

Scalars = Mapping[str, jax.Array]

numDatapoints = 0
datasetName = ''


class TrainState(NamedTuple):
    params: hk.Params
    state: hk.State
    opt_state: optax.OptState
    loss_scale: jmp.LossScale


def get_policy(): return jmp.get_policy(FLAGS.mp_policy)
def get_bn_policy(): return jmp.get_policy(FLAGS.mp_bn_policy)

def nearest2(number):
    return round_to_multiple(number,2)
def round_to_multiple(number, multiple):
    return multiple*round(number/multiple)

def save(ckpt_dir: str, state) -> None:
    with open(os.path.join(ckpt_dir, "arrays.npy"), "wb") as f:
        for x in jax.tree_util.tree_leaves(state):
            np.save(str(f), x, allow_pickle=False)

    tree_struct = jax.tree_map(lambda t: 0, state)
    with open(os.path.join(ckpt_dir, "tree.pkl"), "wb") as f:
        pickle.dump(tree_struct, f)


def restore(ckpt_dir):
    with open(os.path.join(ckpt_dir, "tree.pkl"), "rb") as f:
        tree_struct = pickle.load(f)

    leaves, treedef = jax.tree_flatten(tree_struct)
    with open(os.path.join(ckpt_dir, "arrays.npy"), "rb") as f:
        flat_state = [np.load(str(f)) for _ in leaves]

    return jax.tree_unflatten(treedef, flat_state)


def get_initial_loss_scale() -> jmp.LossScale:
    cls = getattr(jmp, f'{FLAGS.mp_scale_type}LossScale')
    return cls(FLAGS.mp_scale_value) if cls is not jmp.NoOpLossScale else cls()

def _forward(
        batch,
        is_training: bool,
        apply_rng=None
) -> jax.Array:
    """Forward application of the resnet."""
    images = batch['image']

    if is_training:
        apply_rng, aug_rng = jax.random.split(apply_rng)
        images = randomAugmentJax(images,aug_rng,get_policy(),
                                  thruLayer=FLAGS.thruLayer,
                                  onlyLayer=FLAGS.onlyLayer)

    STDDEV_RGB = jnp.array([42.48881573])
    MEAN_RGB = jnp.array([126.52482573])
    images -= MEAN_RGB
    images /= STDDEV_RGB

    net = resnets[FLAGS.resnet_num](FLAGS.layer_size,
                                    resnet_v2=FLAGS.model_resnet_v2,
                                    bn_config={'decay_rate': FLAGS.model_bn_decay})

    mlp = hk.nets.MLP(
        output_sizes=[FLAGS.layer_size, FLAGS.layer_size],
        activate_final=True
    )

    yHold = net(images, is_training=is_training)
    y = hk.BatchNorm(True, True, FLAGS.model_bn_decay)(yHold, is_training=is_training)
    y = mlp(y, FLAGS.dropout_rate, apply_rng) if apply_rng is not None else mlp(y)
    # skip connection
    y = y + yHold
    y = hk.Linear(1)(y)  # No final activation
    return y.flatten()

# Transform our forwards function into a pair of pure functions.
forward = hk.transform_with_state(_forward)

def lr_schedule(step: jax.Array) -> jax.Array:
    """Cosine learning rate schedule."""
    imgSize = imgSizes[FLAGS.img_size_num]
    resNetRelSize = resnetSizes[FLAGS.resnet_num]/50
    batchSize = nearest2(int(FLAGS.batch_size/resNetRelSize*(128/imgSize)**2))
    total_batch_size = batchSize * jax.device_count()
    steps_per_epoch = numDatapoints / total_batch_size
    warmup_steps = FLAGS.train_lr_warmup_epochs * steps_per_epoch
    training_steps = FLAGS.train_epochs * steps_per_epoch

    lr = FLAGS.train_lr_init * batchSize / 256 / jax.device_count()
    scaled_step = (jnp.maximum(step - warmup_steps, 0) /
                   (training_steps - warmup_steps))
    lr *= 0.5 * (1.0 + jnp.cos(jnp.pi * scaled_step))
    if warmup_steps:
        lr *= jnp.minimum(step / warmup_steps, 1.0)
    return lr


def make_optimizer() -> optax.GradientTransformation:
    """SGD with nesterov momentum and a custom lr schedule."""
    return optax.chain(
        optax.trace(
            decay=FLAGS.optimizer_momentum,
            nesterov=FLAGS.optimizer_use_nesterov),
        optax.scale_by_schedule(lr_schedule), optax.scale(-1))


def l2_loss(params: Iterable[jax.Array]) -> jax.Array:
    return 0.5 * sum(jnp.sum(jnp.square(p)) for p in params)


def loss_fn(
        params: hk.Params,
        state: hk.State,
        loss_scale: jmp.LossScale,
        batch,
        rng
) -> Tuple[jax.Array, Tuple[jax.Array, hk.State, jax.Array]]:
    """Computes a regularized loss for the given batch."""
    y, state = forward.apply(params, state, None, batch, is_training=True, apply_rng=rng)
    labels = batch['label']  # jax.nn.one_hot(batch['labels'], 1000)
    mse = ((y - labels) ** 2).mean()
    l2_params = [p for ((mod_name, _), p) in tree.flatten_with_path(params)
                 if 'batchnorm' not in mod_name]
    loss = mse + FLAGS.train_weight_decay * l2_loss(l2_params)

    return loss_scale.scale(loss), (loss, state, mse)


@functools.partial(jax.pmap, axis_name='i', donate_argnums=(0,))
def train_step(
        train_state: TrainState,
        batch,
        rng
) -> Tuple[TrainState, Scalars]:
    """Applies an update to parameters and returns new state."""
    params, state, opt_state, loss_scale = train_state
    grads, (loss, new_state, mse) = (
        jax.grad(loss_fn, has_aux=True)(params, state, loss_scale, batch, rng))

    # Grads are in "param_dtype" (likely F32) here. We cast them back to the
    # compute dtype such that we do the all-reduce below in the compute precision
    # (which is typically lower than the param precision).
    policy = get_policy()
    grads = policy.cast_to_compute(grads)
    grads = loss_scale.unscale(grads)

    # Taking the mean across all replicas to keep params in sync.
    grads = jax.lax.pmean(grads, axis_name='i')

    # We compute our optimizer update in the same precision as params, even when
    # doing mixed precision training.
    grads = policy.cast_to_param(grads)

    # Compute and apply updates via our optimizer.
    updates, new_opt_state = make_optimizer().update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)

    if FLAGS.mp_skip_nonfinite:
        grads_finite = jmp.all_finite(grads)
        loss_scale = loss_scale.adjust(grads_finite)
        new_params, new_state, new_opt_state = jmp.select_tree(
            grads_finite,
            (new_params, new_state, new_opt_state),
            (params, state, opt_state))

    # Scalars to log (note: we log the mean across all hosts/devices).
    scalars = {'train_loss': loss, 'mse': mse, 'loss_scale': loss_scale.loss_scale}
    if FLAGS.mp_skip_nonfinite:
        scalars['grads_finite'] = grads_finite
    new_state, scalars = jmp.cast_to_full((new_state, scalars))
    scalars = jax.lax.pmean(scalars, axis_name='i')
    train_state = TrainState(new_params, new_state, new_opt_state, loss_scale)
    return train_state, scalars


def initial_state(rng: jax.Array, batch, apply_rng) -> TrainState:
    """Computes the initial network state."""
    params, state = forward.init(rng, batch, is_training=True, apply_rng=apply_rng)
    opt_state = make_optimizer().init(params)
    loss_scale = get_initial_loss_scale()
    return TrainState(params, state, opt_state, loss_scale)


# NOTE: We use `jit` not `pmap` here because we want to ensure that we see all
# eval data once and this is not easily satisfiable with pmap (e.g. n=3).
@jax.jit
def eval_batch(
        params: hk.Params,
        state: hk.State,
        batch
) -> Tuple[jax.Array, jax.Array]:
    """Evaluates a batch."""

    y, state = forward.apply(params, state, None, batch, is_training=False)
    labels = batch['label']
    mse = ((y - labels) ** 2).mean()
    return mse, y


# NOTE: We use `jit` not `pmap` here because we want to ensure that we see all
# eval data once and this is not easily satisfiable with pmap (e.g. n=3).
def eval_batch_verbose(
        params: hk.Params,
        state: hk.State,
        batch
) -> Tuple[jax.Array, jax.Array]:
    """Evaluates a batch."""

    y, state = forward.apply(params, state, None, batch, is_training=False)
    labels = batch['label']
    print("___________________")
    print(y)
    print(labels)

    mse = ((y - labels) ** 2)

    return mse.mean(), y

def evaluate(
        params: hk.Params,
        state: hk.State,
) -> Scalars:
    """Evaluates the model at the given params/state."""

    # Params/state are sharded per-device during training. We just need the copy
    # from the first device (since we do not pmap evaluation at the moment).
    params, state = jax.tree_util.tree_map(lambda x: x[0], (params, state))
    eval_dataset = tfds.load(datasetName, split='val')
    eval_dataset = eval_dataset.map(lambda x: (preprocess(x)), num_parallel_calls=-1)
    imgSize = imgSizes[FLAGS.img_size_num]
    resNetRelSize = resnetSizes[FLAGS.resnet_num] / 50
    batchSize = nearest2(int(FLAGS.batch_size/resNetRelSize*(128/imgSize)**2))
    eval_dataset = eval_dataset.batch(batchSize).prefetch(-1)
    eval_dataset = tfds.as_numpy(eval_dataset)

    mseSum = 0
    count = 0

    for batch in eval_dataset:

        if batch['image'].shape[0] == batchSize:
            mse, _ = eval_batch(params, state, batch)
            mseSum += mse
            count += 1

    return {'eval_mse': mseSum / count}


@contextlib.contextmanager
def time_activity(activity_name: str):
    logging.info('[Timing] %s start.', activity_name)
    start = timeit.default_timer()
    yield
    duration = timeit.default_timer() - start
    logging.info('[Timing] %s finished (Took %.2fs).', activity_name, duration)


def test_batch(
        params: hk.Params,
        state: hk.State,
        batch
) -> jax.Array:
    """Evaluates a batch."""

    y, _ = forward.apply(params, state, None, batch, is_training=False)
    return y

def _device_put_sharded(sharded_tree, devices):
  leaves, treedef = jax.tree_util.tree_flatten(sharded_tree)
  n = leaves[0].shape[0]
  return jax.device_put_sharded([
      jax.tree_util.tree_unflatten(treedef, [l[i] for l in leaves])
      for i in range(n)], devices)

def double_buffer(ds):
  """Keeps at least two batches on the accelerator.

  The current GPU allocator design reuses previous allocations. For a training
  loop this means batches will (typically) occupy the same region of memory as
  the previous batch. An issue with this is that it means we cannot overlap a
  host->device copy for the next batch until the previous step has finished and
  the previous batch has been freed.

  By double buffering we ensure that there are always two batches on the device.
  This means that a given batch waits on the N-2'th step to finish and free,
  meaning that it can allocate and copy the next batch to the accelerator in
  parallel with the N-1'th step being executed.

  Args:
    ds: Iterable of batches of numpy arrays.

  Yields:
    Batches of sharded device arrays.
  """
  batch = None
  devices = jax.local_devices()
  for next_batch in ds:
    assert next_batch is not None
    next_batch = _device_put_sharded(next_batch, devices)
    if batch is not None:
      yield batch
    batch = next_batch
  if batch is not None:
    yield batch
def load_dataset(split):
    imgSize = imgSizes[FLAGS.img_size_num]
    resNetRelSize = resnetSizes[FLAGS.resnet_num] / 50
    batchSize = nearest2(int(FLAGS.batch_size/resNetRelSize*(128/imgSize)**2))
    train_dataset = tfds.load(datasetName, split=split)
    num_datapoints = train_dataset.cardinality().numpy()
    train_dataset = train_dataset.repeat().shuffle(1024)

    train_dataset = train_dataset.map(lambda x: (preprocess(x)), num_parallel_calls=-1)

    train_dataset = train_dataset.batch(batchSize).batch(jax.local_device_count()).prefetch(-1)

    train_dataset = tfds.as_numpy(train_dataset)


    if jax.default_backend() == 'gpu':
        train_dataset = double_buffer(train_dataset)
    return train_dataset, num_datapoints

def initAndSummarize(train_dataset, key, subkey):
    # Initialization requires an example input.
    batch = next(train_dataset)
    train_state = jax.pmap(initial_state)(key, batch, subkey)

    # # the key isn't actually used here, it's just for printing; no need to split
    # # Print a useful summary of the execution of our module.
    # summary = hk.experimental.tabulate(train_step)(train_state, batch, key)
    # for line in summary.split('\n'):
    #     logging.info(line)
    return train_state


def setPrecision():
    # Assign mixed precision policies to modules. Note that when training in f16
    # we keep BatchNorm in  full precision. When training with bf16 you can often
    # use bf16 for BatchNorm.
    mp_policy = get_policy()
    bn_policy = get_bn_policy().with_output_dtype(mp_policy.compute_dtype)
    # NOTE: The order we call `set_policy` doesn't matter, when a method on a
    # class is called the policy for that class will be applied, or it will
    # inherit the policy from its parent module.
    hk.mixed_precision.set_policy(hk.BatchNorm, bn_policy)
    hk.mixed_precision.set_policy(hk.nets.ResNet50, mp_policy)


def initKeys():
    # For initialization we need the same random key on each device.
    key = jax.random.PRNGKey(FLAGS.train_init_random_seed)
    key = jnp.broadcast_to(key, (jax.local_device_count(),) + key.shape)
    return splitKeys(key)


def splitKeys(key):
    keys = jax.pmap(jax.random.split)(key)
    return keys[:, 0], keys[:, 1]


# For tensorboard labelling when doing hparam optimization
def getDescriptor():
    imgSize = imgSizes[FLAGS.img_size_num]
    resNetRelSize = resnetSizes[FLAGS.resnet_num] / 50
    batchSize = nearest2(int(FLAGS.batch_size/resNetRelSize*(128/imgSize)**2))
    return (
        'addRandAugment'+
        datasetName + '/' +
        '_' + str(resnets[FLAGS.resnet_num].__name__) +
        '_' + str(imgSize) +
        '_run_num_' + str(FLAGS.run_num)
    )
        # '_thruLayer_' + str(FLAGS.thruLayer) + \
        # '_onlyLayer_' + str(FLAGS.onlyLayer) + \
        # '_bs_' + str(batchSize) + \
        # '_L2_weight_' + str(FLAGS.train_weight_decay) + \
        # '_dropout_' + str(FLAGS.dropout_rate) + \
        # '_learning_rate_' + str(FLAGS.train_lr_init) +\
        # '_bn_decay_' + str(FLAGS.model_bn_decay) + \
        # '_layer_size_' + str(FLAGS.layer_size) + \

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    FLAGS.alsologtostderr = True

    global datasetName
    global numDatapoints
    datasetName = 'age_nih/'

    # Note, you need to use tensorflow-datasets to package the kaggle nih xray dataset into something useable yourself.
    # Seek out the relevant documentation for creating your own tfds datasets.
    # For anyone interested in image processing, tfds is a worthwhile skill, and many learning resources already exist.

    # Example code for creating a tfds dataset can be found in:
    # ./tfdatasets/age_nih/age_nih_dataset_builder.py
    # where it can be run with ``tfds build`` from the terminal in the enclosing directory, ``age_nih``
    train_dataset, numDatapoints = load_dataset('train')
    imgSize = imgSizes[FLAGS.img_size_num]
    resNetRelSize = resnetSizes[FLAGS.resnet_num] / 50
    batchSize = nearest2(int(FLAGS.batch_size/resNetRelSize*(128/imgSize)**2))
    total_train_batch_size = batchSize * jax.device_count()
    num_train_steps = ((numDatapoints * FLAGS.train_epochs) // total_train_batch_size)

    setPrecision()
    key, subkey = initKeys()
    with time_activity("Initializing model"):
        train_state = initAndSummarize(train_dataset, key, subkey)

    descriptor = getDescriptor()
    # Logging for tensorboard
    train_summary_writer = tf.summary.create_file_writer('logs/' + descriptor + '/train')
    val_summary_writer = tf.summary.create_file_writer('logs/' + descriptor + '/val')
    gen_summary_writer = tf.summary.create_file_writer('logs/' + descriptor + '/gen')
    # lastValErr = float('inf')
    print("Beginning train (first step takes a while)")

    minValErr = 1e9
    patience = 100 # for early stopping. note this is typically done on val error rather than train error as seen here.
    minTrainErr = 1e9
    with time_activity('train'):
        for step_num in range(num_train_steps):
            # Take a single training step.
            with jax.profiler.StepTraceAnnotation('train', step_num=step_num):
                batch = next(train_dataset)
                key, subkey = splitKeys(key)
                train_state, train_scalars = train_step(train_state, batch, subkey)

            # Log progress at fixed intervals.
            if step_num and step_num % FLAGS.train_log_every == 0:
                train_scalars = jax.tree_util.tree_map(
                    lambda v: np.mean(v).item(), jax.device_get(train_scalars))
                logging.info('[Train %s/%s] %s',
                             step_num, num_train_steps, np.sqrt(train_scalars['mse']))
                if train_scalars['mse'] < minTrainErr:
                    minTrainErr = train_scalars['mse']
                    patience = 100
                else:
                    patience -= 1
                    if patience == 0:
                        break
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', train_scalars['train_loss'], step=step_num)
                    trainErr = np.sqrt(train_scalars['mse'])
                    tf.summary.scalar('mae', trainErr, step=step_num)

            # By default we do not evaluate during training, but you can configure
            # this with a flag.
            if FLAGS.train_eval_every > 0 and step_num and step_num % FLAGS.train_eval_every == 0:
                with time_activity('eval during train'):
                    eval_scalars = evaluate(train_state.params, train_state.state)
                valErr = np.sqrt(eval_scalars['eval_mse'])
                if valErr < minValErr:
                    minValErr = valErr

                logging.info('[Eval %s/%s] %s', step_num, num_train_steps, valErr)

                with val_summary_writer.as_default():
                    tf.summary.scalar('mae', valErr, step=step_num)
                with gen_summary_writer.as_default():
                    tf.summary.scalar('mae', valErr - trainErr, step=step_num)

    with time_activity('eval final'):
        eval_scalars = evaluate(train_state.params, train_state.state)
    valErr = np.sqrt(eval_scalars['eval_mse'])
    if valErr < minValErr:
        minValErr = valErr

    logging.info('[Eval %s/%s] %s', step_num, num_train_steps, valErr)

    with val_summary_writer.as_default():
        tf.summary.scalar('mae', valErr, step=step_num)
    with gen_summary_writer.as_default():
        tf.summary.scalar('mae', valErr - trainErr, step=step_num)


if __name__ == '__main__':
    gpu.SetGPU(-3, True)
    imgSizes = [128,140,154,170,186,206,226,256,274,302,332,366,402,442,486,512,588,1024]
    resnets = [rn.ResNet4, rn.ResNet6, rn.ResNet8, rn.ResNet10, rn.ResNet18,
               rn.ResNet34, rn.ResNet50, rn.ResNet101, rn.ResNet152, rn.ResNet200]
    resnetSizes = [4,6,8,10,18,34,50,101,152,200]
    app.run(main)
