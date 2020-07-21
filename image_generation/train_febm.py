import tensorflow as tf
import numpy as np
from tensorflow.python.platform import flags

from data import Cifar10
from models import ResNet32, ResNet32Large, ResNet32Larger, ResNet32Wider
from f_divergence import get_divergence_funcs
import os.path as osp
import os
from baselines.logger import TensorBoardOutputFormat
from utils import average_gradients, ReplayBuffer, optimistic_restore
from tqdm import tqdm
import random
from torch.utils.data import DataLoader
import time as time
from io import StringIO
from tensorflow.core.util import event_pb2
import torch
import numpy as np
from custom_adam import AdamOptimizer
from scipy.misc import imsave
import matplotlib.pyplot as plt
from hmc import hmc

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

import horovod.tensorflow as hvd
hvd.init()

from inception import get_inception_score

torch.manual_seed(hvd.rank())
np.random.seed(hvd.rank())
tf.set_random_seed(hvd.rank())

FLAGS = flags.FLAGS


# Dataset Options
flags.DEFINE_string('datasource', 'random',
    'initialization for chains, either random or default (decorruption)')
flags.DEFINE_string('dataset','cifar10', 'dataset to use')
flags.DEFINE_integer('batch_size', 256, 'Size of inputs')
flags.DEFINE_bool('single', False, 'whether to debug by training on a single image')
flags.DEFINE_integer('data_workers', 4,
    'Number of different data workers to load data in parallel')

# General Experiment Settings
flags.DEFINE_string('logdir', 'cachedir',
    'location where log of experiments will be stored')
flags.DEFINE_string('exp', 'default', 'name of experiments')
flags.DEFINE_integer('log_interval', 10, 'log outputs every so many batches')
flags.DEFINE_integer('save_interval', 1000,'save outputs every so many batches')
flags.DEFINE_integer('test_interval', 100,'evaluate outputs every so many batches')
flags.DEFINE_integer('resume_iter', -1, 'iteration to resume training from')
flags.DEFINE_bool('train', True, 'whether to train or test')
flags.DEFINE_integer('epoch_num', 10000, 'Number of Epochs to train on')
flags.DEFINE_float('lr', 3e-4, 'Learning for training')
flags.DEFINE_integer('num_gpus', 1, 'number of gpus to train on')

# f-Divergence
flags.DEFINE_string('divergence', 'reverse_kl', 'which f-divergence to use')

# EBM Specific Experiments Settings
flags.DEFINE_float('ml_coeff', 1.0, 'Maximum Likelihood Coefficients')
flags.DEFINE_float('l2_coeff', 1.0, 'L2 Penalty training')
flags.DEFINE_bool('cclass', False, 'Whether to conditional training in models')
flags.DEFINE_bool('model_cclass', False,'use unsupervised clustering to infer fake labels')
flags.DEFINE_integer('temperature', 1, 'Temperature for energy function')
flags.DEFINE_string('objective', 'cd', 'use either contrastive divergence objective(least stable),'
                    'logsumexp(more stable)'
                    'softplus(most stable)')
flags.DEFINE_bool('zero_kl', False, 'whether to zero out the kl loss')

# Setting for MCMC sampling
flags.DEFINE_float('proj_norm', 0.0, 'Maximum change of input images')
flags.DEFINE_string('proj_norm_type', 'li', 'Either li or l2 ball projection')
flags.DEFINE_integer('num_steps', 20, 'Steps of gradient descent for training')
flags.DEFINE_float('step_lr', 1.0, 'Size of steps for gradient descent')
flags.DEFINE_bool('replay_batch', False, 'Use MCMC chains initialized from a replay buffer.')
flags.DEFINE_bool('hmc', False, 'Whether to use HMC sampling to train models')
flags.DEFINE_float('noise_scale', 1.,'Relative amount of noise for MCMC')
flags.DEFINE_bool('pcd', False, 'whether to use pcd training instead')

# Architecture Settings
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets')
flags.DEFINE_bool('spec_norm', True, 'Whether to use spectral normalization on weights')
flags.DEFINE_bool('use_attention', False, 'Whether to use self attention in network')
flags.DEFINE_bool('large_model', False, 'whether to use a large model')
flags.DEFINE_bool('larger_model', False, 'Deeper ResNet32 Network')
flags.DEFINE_bool('wider_model', False, 'Wider ResNet32 Network')

# Dataset settings
flags.DEFINE_bool('mixup', False, 'whether to add mixup to training images')
flags.DEFINE_bool('augment', False, 'whether to augmentations to images')
flags.DEFINE_float('rescale', 1.0, 'Factor to rescale inputs from 0-1 box')

# Dsprites specific experiments
flags.DEFINE_bool('cond_shape', False, 'condition of shape type')
flags.DEFINE_bool('cond_size', False, 'condition of shape size')
flags.DEFINE_bool('cond_pos', False, 'condition of position loc')
flags.DEFINE_bool('cond_rot', False, 'condition of rot')

FLAGS.step_lr = FLAGS.step_lr * FLAGS.rescale

FLAGS.batch_size *= FLAGS.num_gpus

print("{} batch size".format(FLAGS.batch_size))


def compress_x_mod(x_mod):
    x_mod = (255 * np.clip(x_mod, 0, FLAGS.rescale) / FLAGS.rescale).astype(np.uint8)
    return x_mod


def decompress_x_mod(x_mod):
    x_mod = x_mod / 256 * FLAGS.rescale + \
        np.random.uniform(0, 1 / 256 * FLAGS.rescale, x_mod.shape)
    return x_mod


def make_image(tensor):
    """Convert an numpy representation image to Image protobuf"""
    from PIL import Image
    if len(tensor.shape) == 4:
        _, height, width, channel = tensor.shape
    elif len(tensor.shape) == 3:
        height, width, channel = tensor.shape
    elif len(tensor.shape) == 2:
        height, width = tensor.shape
        channel = 1
    tensor = tensor.astype(np.uint8)
    image = Image.fromarray(tensor)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)


def log_image(im, logger, tag, step=0):
    im = make_image(im)

    summary = [tf.Summary.Value(tag=tag, image=im)]
    summary = tf.Summary(value=summary)
    event = event_pb2.Event(summary=summary)
    event.step = step
    logger.writer.WriteEvent(event)
    logger.writer.Flush()


def rescale_im(image):
    image = np.clip(image, 0, FLAGS.rescale)
    if FLAGS.dataset == 'mnist' or FLAGS.dataset == 'dsprites':
        return (np.clip((FLAGS.rescale - image) * 256 / FLAGS.rescale, 0, 255)).astype(np.uint8)
    else:
        return (np.clip(image * 256 / FLAGS.rescale, 0, 255)).astype(np.uint8)


def train(target_vars, saver, sess, logger, dataloader, resume_iter, logdir):
    X = target_vars['X']
    Y = target_vars['Y']
    X_NOISE = target_vars['X_NOISE']
    train_op_model = target_vars['train_op_model']
    train_op_dis = target_vars['train_op_dis']
    energy_pos = target_vars['energy_pos']
    energy_neg = target_vars['energy_neg']
    score_pos = target_vars['score_pos']
    score_neg = target_vars['score_neg']
    loss_energy = target_vars['loss_energy']
    loss_total = target_vars['total_loss']
    gvs = target_vars['gvs']
    x_grad = target_vars['x_grad']
    x_grad_first = target_vars['x_grad_first']
    x_off = target_vars['x_off']
    temp = target_vars['temp']
    x_mod = target_vars['x_mod']
    LABEL = target_vars['LABEL']
    LABEL_POS = target_vars['LABEL_POS']
    weights = target_vars['weights']
    test_x_mod = target_vars['test_x_mod']
    eps = target_vars['eps_begin']
    label_ent = target_vars['label_ent']
    train_op_model_l2 = target_vars['train_op_model_l2']
    train_op_dis_l2 = target_vars['train_op_dis_l2']

    output = [train_op_model, x_mod]

    if FLAGS.use_attention:
        gamma = weights[0]['atten']['gamma']
    else:
        gamma = tf.zeros(1)

    val_output = [test_x_mod]

    gvs_dict = dict(gvs)

    # log_output = [
    #     train_op,
    #     energy_pos,
    #     energy_neg,
    #     eps,
    #     loss_energy,
    #     loss_total,
    #     x_grad,
    #     x_off,
    #     x_mod,
    #     gamma,
    #     x_grad_first,
    #     label_ent,
    #     *gvs_dict.keys()]

    replay_buffer = ReplayBuffer(10000)
    itr = resume_iter
    x_mod = None
    gd_steps = 1

    dataloader_iterator = iter(dataloader)
    best_inception = 0.0
    save_interval = FLAGS.save_interval

    for epoch in range(FLAGS.epoch_num):
        for data_corrupt, data, label in dataloader:
            data_corrupt = data_corrupt_init = data_corrupt.numpy()
            data_corrupt_init = data_corrupt.copy()

            data = data.numpy()
            label = label.numpy()

            label_init = label.copy()

            if FLAGS.mixup:
                idx = np.random.permutation(data.shape[0])
                lam = np.random.beta(1, 1, size=(data.shape[0], 1, 1, 1))
                data = data * lam + data[idx] * (1 - lam)

            if FLAGS.replay_batch and (x_mod is not None):
                replay_buffer.add(compress_x_mod(x_mod))

                if len(replay_buffer) > FLAGS.batch_size:
                    replay_batch = replay_buffer.sample(FLAGS.batch_size)
                    replay_batch = decompress_x_mod(replay_batch)
                    replay_mask = (
                        np.random.uniform(
                            0,
                            FLAGS.rescale,
                            FLAGS.batch_size) > 0.05)
                    data_corrupt[replay_mask] = replay_batch[replay_mask]

            if FLAGS.pcd:
                if x_mod is not None:
                    data_corrupt = x_mod

            feed_dict = {X_NOISE: data_corrupt, X: data, Y: label}

            if FLAGS.cclass:
                feed_dict[LABEL] = label
                feed_dict[LABEL_POS] = label_init

            if itr > 10:
                # Train discriminator
                _ = sess.run(train_op_dis, feed_dict)

                # Train model
                _, x_mod = sess.run(output, feed_dict)
            else:
                _, _ = sess.run([train_op_dis_l2, train_op_model_l2], feed_dict)
                energy_neg_, energy_pos_, score_neg_, score_pos_ = sess.run([energy_neg, energy_pos, score_neg, score_pos], feed_dict)
                print(np.mean(energy_neg_), np.mean(energy_pos_), np.mean(score_neg_), np.mean(score_pos_))

            if itr > 30000:
                save_interval = 100

            # if itr % save_interval == 0 and hvd.rank() == 0:
            #     saver.save(sess, osp.join(FLAGS.logdir, FLAGS.exp, 'model_{}'.format(itr)))

            if itr and itr % FLAGS.test_interval == 0 and hvd.rank() == 0 and FLAGS.dataset != '2d':
                try_im = x_mod
                orig_im = data_corrupt.squeeze()
                actual_im = rescale_im(data)

                orig_im = rescale_im(orig_im)
                try_im = rescale_im(try_im).squeeze()

                for i, (im, t_im, actual_im_i) in enumerate(
                        zip(orig_im[:20], try_im[:20], actual_im)):
                    shape = orig_im.shape[1:]
                    new_im = np.zeros((shape[0], shape[1] * 3, *shape[2:]))
                    size = shape[1]
                    new_im[:, :size] = im
                    new_im[:, size:2 * size] = t_im
                    new_im[:, 2 * size:] = actual_im_i

                    log_image(
                        new_im, logger, 'train_gen_{}'.format(itr), step=i)

                test_im = x_mod

                try:
                    data_corrupt, data, label = next(dataloader_iterator)
                except BaseException:
                    dataloader_iterator = iter(dataloader)
                    data_corrupt, data, label = next(dataloader_iterator)

                data_corrupt = data_corrupt.numpy()

                if FLAGS.replay_batch and (
                        x_mod is not None) and len(replay_buffer) > 0:
                    replay_batch = replay_buffer.sample(FLAGS.batch_size)
                    replay_batch = decompress_x_mod(replay_batch)
                    replay_mask = (
                        np.random.uniform(
                            0, 1, (FLAGS.batch_size)) > 0.05)
                    data_corrupt[replay_mask] = replay_batch[replay_mask]

                if FLAGS.dataset == 'cifar10' or FLAGS.dataset == 'imagenet' or FLAGS.dataset == 'imagenetfull' or FLAGS.dataset == 'celeba':
                    n = 128

                    if FLAGS.dataset == "imagenetfull":
                        n = 32

                    if len(replay_buffer) > n:
                        data_corrupt = decompress_x_mod(replay_buffer.sample(n))
                    elif FLAGS.dataset == 'imagenetfull':
                        data_corrupt = np.random.uniform(
                            0, FLAGS.rescale, (n, 128, 128, 3))
                    else:
                        data_corrupt = np.random.uniform(
                            0, FLAGS.rescale, (n, 32, 32, 3))

                    if FLAGS.dataset == 'cifar10':
                        label = np.eye(10)[np.random.randint(0, 10, (n))]
                    elif FLAGS.dataset == 'celeba':
                        label = np.array([1] * n). reshape((n, 1))
                    else:
                        label = np.eye(1000)[
                            np.random.randint(
                                0, 1000, (n))]

                feed_dict[X_NOISE] = data_corrupt

                feed_dict[X] = data

                if FLAGS.cclass:
                    feed_dict[LABEL] = label

                test_x_mod = sess.run(val_output, feed_dict)

                try_im = test_x_mod
                orig_im = data_corrupt.squeeze()
                actual_im = rescale_im(data.numpy())

                orig_im = rescale_im(orig_im)
                try_im = rescale_im(try_im).squeeze()

                for i, (im, t_im, actual_im_i) in enumerate(
                        zip(orig_im[:20], try_im[:20], actual_im)):

                    shape = orig_im.shape[1:]
                    new_im = np.zeros((shape[0], shape[1] * 3, *shape[2:]))
                    size = shape[1]
                    new_im[:, :size] = im
                    new_im[:, size:2 * size] = t_im
                    new_im[:, 2 * size:] = actual_im_i
                    log_image(
                        new_im, logger, 'val_gen_{}'.format(itr), step=i)

                score, std = get_inception_score(list(try_im), splits=1)
                print("Iteration {}: Inception score of {} with std of {}".format(itr, score, std))
                kvs = {}
                kvs['inception_score'] = score
                kvs['inception_score_std'] = std
                logger.writekvs(kvs)

                if score > best_inception:
                    best_inception = score
                    saver.save(sess, osp.join(FLAGS.logdir, FLAGS.exp, 'model_best'))
                    saver.save(sess, osp.join(FLAGS.logdir, FLAGS.exp, 'model_{}'.format(itr)))

            if itr > 60000 and FLAGS.dataset == "mnist":
                assert False
            itr += 1

    saver.save(sess, osp.join(FLAGS.logdir, FLAGS.exp, 'model_{}'.format(itr)))


cifar10_map = {0: 'airplane',
               1: 'automobile',
               2: 'bird',
               3: 'cat',
               4: 'deer',
               5: 'dog',
               6: 'frog',
               7: 'horse',
               8: 'ship',
               9: 'truck'}


def test(target_vars, saver, sess, logger, dataloader):
    X_NOISE = target_vars['X_NOISE']
    X = target_vars['X']
    Y = target_vars['Y']
    LABEL = target_vars['LABEL']
    energy_start = target_vars['energy_start']
    x_mod = target_vars['x_mod']
    x_mod = target_vars['test_x_mod']
    energy_neg = target_vars['energy_neg']

    np.random.seed(1)
    random.seed(1)

    output = [x_mod, energy_start, energy_neg]

    dataloader_iterator = iter(dataloader)
    data_corrupt, data, label = next(dataloader_iterator)
    data_corrupt, data, label = data_corrupt.numpy(), data.numpy(), label.numpy()

    orig_im = try_im = data_corrupt

    if FLAGS.cclass:
        try_im, energy_orig, energy = sess.run(
            output, {X_NOISE: orig_im, Y: label[0:1], LABEL: label})
    else:
        try_im, energy_orig, energy = sess.run(
            output, {X_NOISE: orig_im, Y: label[0:1]})

    orig_im = rescale_im(orig_im)
    try_im = rescale_im(try_im)
    actual_im = rescale_im(data)

    for i, (im, energy_i, t_im, energy, label_i, actual_im_i) in enumerate(
            zip(orig_im, energy_orig, try_im, energy, label, actual_im)):
        label_i = np.array(label_i)

        shape = im.shape[1:]
        new_im = np.zeros((shape[0], shape[1] * 3, *shape[2:]))
        size = shape[1]
        new_im[:, :size] = im
        new_im[:, size:2 * size] = t_im

        if FLAGS.cclass:
            label_i = np.where(label_i == 1)[0][0]
            if FLAGS.dataset == 'cifar10':
                log_image(new_im, logger, '{}_{:.4f}_now_{:.4f}_{}'.format(
                    i, energy_i[0], energy[0], cifar10_map[label_i]), step=i)
            else:
                log_image(
                    new_im,
                    logger,
                    '{}_{:.4f}_now_{:.4f}_{}'.format(
                        i,
                        energy_i[0],
                        energy[0],
                        label_i),
                    step=i)
        else:
            log_image(
                new_im,
                logger,
                '{}_{:.4f}_now_{:.4f}'.format(
                    i,
                    energy_i[0],
                    energy[0]),
                step=i)

    test_ims = list(try_im)
    real_ims = list(actual_im)

    for i in tqdm(range(50000 // FLAGS.batch_size + 1)):
        try:
            data_corrupt, data, label = dataloader_iterator.next()
        except BaseException:
            dataloader_iterator = iter(dataloader)
            data_corrupt, data, label = dataloader_iterator.next()

        data_corrupt, data, label = data_corrupt.numpy(), data.numpy(), label.numpy()

        if FLAGS.cclass:
            try_im, energy_orig, energy = sess.run(
                output, {X_NOISE: data_corrupt, Y: label[0:1], LABEL: label})
        else:
            try_im, energy_orig, energy = sess.run(
                output, {X_NOISE: data_corrupt, Y: label[0:1]})

        try_im = rescale_im(try_im)
        real_im = rescale_im(data)

        test_ims.extend(list(try_im))
        real_ims.extend(list(real_im))

    score, std = get_inception_score(test_ims)
    print("Inception score of {} with std of {}".format(score, std))


def log_mean_exp(inputs):
    s = tf.reduce_max(inputs)
    return s + tf.log(tf.reduce_mean(tf.exp(inputs - s)))


def np_log_mean_exp(inputs):
    s = np.max(inputs)
    return s + np.log(np.mean(np.exp(inputs - s)))


def main():
    print("Local rank: ", hvd.local_rank(), hvd.size())
    FLAGS.exp = FLAGS.exp + '_' + FLAGS.divergence

    logdir = osp.join(FLAGS.logdir, FLAGS.exp)
    if hvd.rank() == 0:
        if not osp.exists(logdir):
            os.makedirs(logdir)
        logger = TensorBoardOutputFormat(logdir)
    else:
        logger = None

    print("Loading data...")
    dataset = Cifar10(augment=FLAGS.augment, rescale=FLAGS.rescale)
    test_dataset = Cifar10(train=False, rescale=FLAGS.rescale)
    channel_num = 3

    X_NOISE = tf.placeholder(shape=(None, 32, 32, 3), dtype=tf.float32)
    X = tf.placeholder(shape=(None, 32, 32, 3), dtype=tf.float32)
    LABEL = tf.placeholder(shape=(None, 10), dtype=tf.float32)
    LABEL_POS = tf.placeholder(shape=(None, 10), dtype=tf.float32)

    if FLAGS.large_model:
        model = ResNet32Large(
            num_channels=channel_num,
            num_filters=128,
            train=True)
        model_dis = ResNet32Large(
            num_channels=channel_num,
            num_filters=128,
            train=True)
    elif FLAGS.larger_model:
        model = ResNet32Larger(
            num_channels=channel_num,
            num_filters=128)
        model_dis = ResNet32Larger(
            num_channels=channel_num,
            num_filters=128)
    elif FLAGS.wider_model:
        model = ResNet32Wider(
            num_channels=channel_num,
            num_filters=256)
        model_dis = ResNet32Wider(
            num_channels=channel_num,
            num_filters=256)
    else:
        model = ResNet32(
            num_channels=channel_num,
            num_filters=128)
        model_dis = ResNet32(
            num_channels=channel_num,
            num_filters=128)

    print("Done loading...")

    grad_exp, conjugate_grad_exp = get_divergence_funcs(FLAGS.divergence)

    data_loader = DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.data_workers,
        drop_last=True,
        shuffle=True)

    weights = [model.construct_weights('context_energy'), model_dis.construct_weights('context_dis')]

    Y = tf.placeholder(shape=(None), dtype=tf.int32)

    # Varibles to run in training
    X_SPLIT = tf.split(X, FLAGS.num_gpus)
    X_NOISE_SPLIT = tf.split(X_NOISE, FLAGS.num_gpus)
    LABEL_SPLIT = tf.split(LABEL, FLAGS.num_gpus)
    LABEL_POS_SPLIT = tf.split(LABEL_POS, FLAGS.num_gpus)
    LABEL_SPLIT_INIT = list(LABEL_SPLIT)
    tower_grads = []
    tower_grads_dis = []
    tower_grads_l2 = []
    tower_grads_dis_l2 = []

    optimizer = AdamOptimizer(FLAGS.lr, beta1=0.0, beta2=0.999)
    optimizer = hvd.DistributedOptimizer(optimizer)

    optimizer_dis = AdamOptimizer(FLAGS.lr, beta1=0.0, beta2=0.999)
    optimizer_dis = hvd.DistributedOptimizer(optimizer_dis)

    for j in range(FLAGS.num_gpus):

        energy_pos = [
            model.forward(
                X_SPLIT[j],
                weights[0],
                label=LABEL_POS_SPLIT[j],
                stop_at_grad=False)]
        energy_pos = tf.concat(energy_pos, axis=0)

        score_pos = [
            model_dis.forward(
                X_SPLIT[j],
                weights[1],
                label=LABEL_POS_SPLIT[j],
                stop_at_grad=False)]
        score_pos = tf.concat(score_pos, axis=0)

        print("Building graph...")
        x_mod = x_orig = X_NOISE_SPLIT[j]

        x_grads = []

        energy_negs = []
        loss_energys = []

        energy_negs.extend([model.forward(tf.stop_gradient(
            x_mod), weights[0], label=LABEL_SPLIT[j], stop_at_grad=False, reuse=True)])
        eps_begin = tf.zeros(1)

        steps = tf.constant(0)
        c = lambda i, x: tf.less(i, FLAGS.num_steps)

        def langevin_step(counter, x_mod):
            x_mod = x_mod + tf.random_normal(tf.shape(x_mod),
                                             mean=0.0,
                                             stddev=0.005 * FLAGS.rescale * FLAGS.noise_scale)

            energy_noise = energy_start = tf.concat(
                [model.forward(
                        x_mod,
                        weights[0],
                        label=LABEL_SPLIT[j],
                        reuse=True,
                        stop_at_grad=False,
                        stop_batch=True)],
                axis=0)

            x_grad, label_grad = tf.gradients(energy_noise, [x_mod, LABEL_SPLIT[j]])
            energy_noise_old = energy_noise

            lr = FLAGS.step_lr

            if FLAGS.proj_norm != 0.0:
                if FLAGS.proj_norm_type == 'l2':
                    x_grad = tf.clip_by_norm(x_grad, FLAGS.proj_norm)
                elif FLAGS.proj_norm_type == 'li':
                    x_grad = tf.clip_by_value(
                        x_grad, -FLAGS.proj_norm, FLAGS.proj_norm)
                else:
                    print("Other types of projection are not supported!!!")
                    assert False

            # Clip gradient norm for now
            if FLAGS.hmc:
                # Step size should be tuned to get around 65% acceptance
                def energy(x):
                    return FLAGS.temperature * \
                        model.forward(x, weights[0], label=LABEL_SPLIT[j], reuse=True)

                x_last = hmc(x_mod, 15., 10, energy)
            else:
                x_last = x_mod - (lr) * x_grad

            x_mod = x_last
            x_mod = tf.clip_by_value(x_mod, 0, FLAGS.rescale)

            counter = counter + 1

            return counter, x_mod

        steps, x_mod = tf.while_loop(c, langevin_step, (steps, x_mod))

        energy_eval = model.forward(x_mod, weights[0], label=LABEL_SPLIT[j],
                                    stop_at_grad=False, reuse=True)
        x_grad = tf.gradients(energy_eval, [x_mod])[0]
        x_grads.append(x_grad)

        energy_negs.append(
            model.forward(
                tf.stop_gradient(x_mod),
                weights[0],
                label=LABEL_SPLIT[j],
                stop_at_grad=False,
                reuse=True))

        score_neg = model_dis.forward(
                tf.stop_gradient(x_mod),
                weights[1],
                label=LABEL_SPLIT[j],
                stop_at_grad=False,
                reuse=True)

        test_x_mod = x_mod

        temp = FLAGS.temperature

        energy_neg = energy_negs[-1]
        x_off = tf.reduce_mean(
            tf.abs(x_mod[:tf.shape(X_SPLIT[j])[0]] - X_SPLIT[j]))

        loss_energy = model.forward(
            x_mod,
            weights[0],
            reuse=True,
            label=LABEL,
            stop_grad=True)

        print("Finished processing loop construction ...")

        target_vars = {}

        if FLAGS.cclass or FLAGS.model_cclass:
            label_sum = tf.reduce_sum(LABEL_SPLIT[0], axis=0)
            label_prob = label_sum / tf.reduce_sum(label_sum)
            label_ent = -tf.reduce_sum(label_prob *
                                       tf.math.log(label_prob + 1e-7))
        else:
            label_ent = tf.zeros(1)

        target_vars['label_ent'] = label_ent

        if FLAGS.train:

            loss_dis = - (tf.reduce_mean(grad_exp(score_pos + energy_pos)) - tf.reduce_mean(conjugate_grad_exp(score_neg + energy_neg)))
            loss_dis = loss_dis + FLAGS.l2_coeff * (tf.reduce_mean(tf.square(score_pos)) + tf.reduce_mean(tf.square(score_neg)))
            l2_dis = FLAGS.l2_coeff * (tf.reduce_mean(tf.square(score_pos)) + tf.reduce_mean(tf.square(score_neg)))

            loss_model = tf.reduce_mean(grad_exp(score_pos + energy_pos)) + \
                         tf.reduce_mean(energy_neg * tf.stop_gradient(conjugate_grad_exp(score_neg + energy_neg))) - \
                         tf.reduce_mean(energy_neg) * tf.stop_gradient(tf.reduce_mean(conjugate_grad_exp(score_neg + energy_neg))) - \
                         tf.reduce_mean(conjugate_grad_exp(score_neg + energy_neg))
            loss_model = loss_model + FLAGS.l2_coeff * (tf.reduce_mean(tf.square(energy_pos)) + tf.reduce_mean(tf.square(energy_neg)))
            l2_model = FLAGS.l2_coeff * (tf.reduce_mean(tf.square(energy_pos)) + tf.reduce_mean(tf.square(energy_neg)))

            print("Started gradient computation...")
            model_vars = [var for var in tf.trainable_variables() if 'context_energy' in var.name]
            print("model var number", len(model_vars))
            dis_vars = [var for var in tf.trainable_variables() if 'context_dis' in var.name]
            print("discriminator var number", len(dis_vars))

            gvs = optimizer.compute_gradients(loss_model, model_vars)
            gvs = [(k, v) for (k, v) in gvs if k is not None]
            tower_grads.append(gvs)

            gvs = optimizer.compute_gradients(l2_model, model_vars)
            gvs = [(k, v) for (k, v) in gvs if k is not None]
            tower_grads_l2.append(gvs)

            gvs_dis = optimizer_dis.compute_gradients(loss_dis, dis_vars)
            gvs_dis = [(k, v) for (k, v) in gvs_dis if k is not None]
            tower_grads_dis.append(gvs_dis)

            gvs_dis = optimizer_dis.compute_gradients(l2_dis, dis_vars)
            gvs_dis = [(k, v) for (k, v) in gvs_dis if k is not None]
            tower_grads_dis_l2.append(gvs_dis)

            print("Finished applying gradients.")

            target_vars['total_loss'] = loss_model
            target_vars['loss_energy'] = loss_energy
            target_vars['weights'] = weights
            target_vars['gvs'] = gvs

        target_vars['X'] = X
        target_vars['Y'] = Y
        target_vars['LABEL'] = LABEL
        target_vars['LABEL_POS'] = LABEL_POS
        target_vars['X_NOISE'] = X_NOISE
        target_vars['energy_pos'] = energy_pos
        target_vars['energy_start'] = energy_negs[0]

        if len(x_grads) >= 1:
            target_vars['x_grad'] = x_grads[-1]
            target_vars['x_grad_first'] = x_grads[0]
        else:
            target_vars['x_grad'] = tf.zeros(1)
            target_vars['x_grad_first'] = tf.zeros(1)

        target_vars['x_mod'] = x_mod
        target_vars['x_off'] = x_off
        target_vars['temp'] = temp
        target_vars['energy_neg'] = energy_neg
        target_vars['test_x_mod'] = test_x_mod
        target_vars['eps_begin'] = eps_begin
        target_vars['score_neg'] = score_neg
        target_vars['score_pos'] = score_pos

    if FLAGS.train:
        grads_model = average_gradients(tower_grads)
        train_op_model = optimizer.apply_gradients(grads_model)
        target_vars['train_op_model'] = train_op_model

        grads_model_l2 = average_gradients(tower_grads_l2)
        train_op_model_l2 = optimizer.apply_gradients(grads_model_l2)
        target_vars['train_op_model_l2'] = train_op_model_l2

        grads_model_dis = average_gradients(tower_grads_dis)
        train_op_dis = optimizer_dis.apply_gradients(grads_model_dis)
        target_vars['train_op_dis'] = train_op_dis

        grads_model_dis_l2 = average_gradients(tower_grads_dis_l2)
        train_op_dis_l2 = optimizer_dis.apply_gradients(grads_model_dis_l2)
        target_vars['train_op_dis_l2'] = train_op_dis_l2

    config = tf.ConfigProto()

    if hvd.size() > 1:
        config.gpu_options.visible_device_list = str(hvd.local_rank())

    sess = tf.Session(config=config)

    saver = loader = tf.train.Saver(max_to_keep=500)

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("Model has a total of {} parameters".format(total_parameters))

    sess.run(tf.global_variables_initializer())

    resume_itr = 0

    if (FLAGS.resume_iter != -1 or not FLAGS.train) and hvd.rank() == 0:
        model_file = osp.join(logdir, 'model_{}'.format(FLAGS.resume_iter))
        resume_itr = FLAGS.resume_iter
        saver.restore(sess, model_file)
        # optimistic_restore(sess, model_file)

    sess.run(hvd.broadcast_global_variables(0))
    print("Initializing variables...")

    print("Start broadcast")
    print("End broadcast")

    if FLAGS.train:
        train(target_vars, saver, sess,
              logger, data_loader, resume_itr,
              logdir)

    test(target_vars, saver, sess, logger, data_loader)


if __name__ == "__main__":
    main()
