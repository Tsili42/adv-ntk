import pickle
import jax
from jax import numpy as jnp
from jax import scipy as jsp
from matplotlib import pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds
import neural_tangents as nt
import os

import numpy as np
from adv_utils import _fast_gradient_method
from flax import linen as nn

import matplotlib as mpl

cmap = mpl.cm.plasma
EPOCHS = 200+1
BATCHSIZE = 250
DATASET = 'cifar10'
s = 5
n = EPOCHS
T = np.linspace(0, 1, n)

def kernel_similarity(K1, K2):
    d11, d12, d22 = jnp.trace(K1 @ K1.T), jnp.trace(K1 @ K2.T), jnp.trace(K2 @ K2.T)

    return d12 / ((d11)**(1/2) * (d22)**(1/2))

def compute_distance_train(kernel_path, binary=False):

    # 201 is the number of epochs (200 + init)
    sim = jnp.zeros((EPOCHS, EPOCHS))
    if (not binary):
        for i in range(EPOCHS):
            for j in range(i+1):
                K1 = jnp.load(kernel_path+f'/kernel_epoch_{i}.npy').transpose([3, 1, 2, 0]).reshape(BATCHSIZE*10, BATCHSIZE*10)
                K2 = jnp.load(kernel_path+f'/kernel_epoch_{j}.npy').transpose([3, 1, 2, 0]).reshape(BATCHSIZE*10, BATCHSIZE*10)
                sim = sim.at[i, j].set(kernel_similarity(K1, K2))
    else:
        for i in range(EPOCHS):
            for j in range(i+1):
                K1 = jnp.squeeze(jnp.load(kernel_path+f'/kernel_epoch_{i}.npy'))
                K2 = jnp.squeeze(jnp.load(kernel_path+f'/kernel_epoch_{j}.npy'))
                sim = sim.at[i, j].set(kernel_similarity(K1, K2))

    return 1 - jnp.maximum(sim, sim.T)

def compute_polar_dynamics(kernel_path, space=None, normalized=True, binary=False, top=True):
    '''
    kernel_path: prefix of path to search for kernels

    space, int: principal space to keep track of. If None, tracks the movement of the whole space.

    normalized: whether to normalize the radius of the movement to [0, 1].

    binary: takes care of the shape of the kernel - no permutation of the axis.

    top: whether space corresponds to the top ``space" eigenvalues, or the bottom.
    '''
    if (binary):
        init = jnp.squeeze(jnp.load(kernel_path+f'/kernel_epoch_0.npy'))
        final = jnp.squeeze(jnp.load(kernel_path+f'/kernel_epoch_200.npy'))
    else:
        init = jnp.load(kernel_path+f'/kernel_epoch_0.npy').transpose([3, 1, 2, 0]).reshape(BATCHSIZE*10, BATCHSIZE*10)
        final = jnp.load(kernel_path+f'/kernel_epoch_200.npy').transpose([3, 1, 2, 0]).reshape(BATCHSIZE*10, BATCHSIZE*10)
    if (space is not None):
        w, v = jsp.linalg.eigh(init)
        if (top):
            init = v[:, -space:] @ jnp.diag(w[-space:]) @ v.T[-space:, :]
        else:
            # bottom eig space
            init = v[:, :space] @ jnp.diag(w[:space]) @ v.T[:space, :]

        w, v = jsp.linalg.eigh(final)
        if (top):
            final = v[:, -space:] @ jnp.diag(w[-space:]) @ v.T[-space:, :]
        else:
            final = v[:, :space] @ jnp.diag(w[:space]) @ v.T[:space, :]

    final_norm = jnp.linalg.norm(final - init)

    step = 1
    radius, angle = [0], [0]
    for i in range(1, EPOCHS, step):
        if (binary):
            current_kernel = jnp.squeeze(jnp.load(kernel_path+f'/kernel_epoch_{i}.npy'))
        else:
            current_kernel = jnp.load(kernel_path+f'/kernel_epoch_{i}.npy').transpose([3, 1, 2, 0]).reshape(BATCHSIZE*10, BATCHSIZE*10)
        if (space is not None):
            w, v = jsp.linalg.eigh(current_kernel)
            if (top):
                current_kernel = v[:, -space:] @ jnp.diag(w[-space:]) @ v.T[-space:, :]
            else:
                current_kernel = v[:, :space] @ jnp.diag(w[:space]) @ v.T[:space, :]
        radius.append(jnp.linalg.norm(current_kernel - init))
        angle.append(jnp.arccos(kernel_similarity(current_kernel, init)))

    if (normalized):
        return [i / final_norm for i in radius], angle
    else:
        return [i for i in radius], angle

def compute_top_mass(kernel_path, step=1, binary=False):

    mass = {}
    mass['top_10'], mass['top_20'] = [], []
    for i in range(0, EPOCHS, step):
        if (binary):
            cur = jnp.squeeze(jnp.load(kernel_path+f'/kernel_epoch_{i}.npy'))
        else:
            cur = jnp.load(kernel_path+f'/kernel_epoch_{i}.npy').transpose([3, 1, 2, 0]).reshape(BATCHSIZE*10, BATCHSIZE*10)
        l = jsp.linalg.eigh(cur, eigvals_only=True)
        mass['top_10'].append(jnp.sum(jax.lax.map(lambda x: x**2, l[-10:])) / jnp.sum(jax.lax.map(lambda x: x**2, l)))
        mass['top_20'].append(jnp.sum(jax.lax.map(lambda x: x**2, l[-20:])) / jnp.sum(jax.lax.map(lambda x: x**2, l)))

    return mass


paths = [
    f'./sgd_robust_train_results/{DATASET}/clean_train/kernels',
    f'./sgd_robust_train_results/{DATASET}/adv_train1/kernels',
    f'./sgd_robust_train_results/{DATASET}/adv_train20/kernels'
    # './robust_train_results/cifar10binary_PGD10/adv_train/kernels'
]

name_dict = {
    0:  'clean',
    1:  'fgsm',
    2:  'pgd_20'
    # 3:  'binary_pgd10'
}
# paths = [
#     './robust_train_results/cifar10binary_PGD10/adv_train/kernels'
# ]
s = 5
n = EPOCHS
T = np.linspace(0, 1, n)
for i, kernel_path in enumerate(paths):

    # Distance heatmap
    dist = compute_distance_train(kernel_path)
    jnp.save(kernel_path+'/distance_during_training', dist)
    # # dist = jnp.load(kernel_path+'/distance_during_training.npy')
    dist_flipped = jnp.flip(dist, axis=0) # for visualization purposes
    plt.imshow(dist_flipped, cmap='tab20c')
    plt.title('Kernel distance')
    plt.colorbar()
    plt.yticks(ticks=[0, 50, 100, 150, 200], labels=[200, 150, 100, 50, 0])
    # plt.savefig(f'./sgd_measurements/{DATASET}/distance_{name_dict[i]}.pdf')
    plt.show()
    
    # Polar phase diagram
    radius, angle = compute_polar_dynamics(kernel_path)
    jnp.save(kernel_path+'/radius', radius)
    jnp.save(kernel_path+'/angle', angle)
    # # radius, angle = jnp.load(kernel_path+'/radius.npy'), jnp.load(kernel_path+'/angle.npy')
    fig = plt.figure()
    ax1 = fig.add_axes([0.1,0.1,0.8,0.8], polar=True)
    # ax1.plot(angle, radius, lw=2.5, label='std')
    for j in range(0, n-s, s):
        ax1.plot(angle[j:j+s+1], radius[j:j+s+1], color=cmap(T[j]))
    # ax1.plot(angle, radius, lw=2.5)
    ax1.set_xlabel('radius')
    ax1.set_ylabel('angle')
    ax1.set_xlim(0,1)
    ax1.set_title('Polar dynamics')
    # plt.savefig(f'./sgd_measurements/{DATASET}/polar_{name_dict[i]}.pdf')
    plt.show()
    plt.close()
    
    # Polar phase diagram for top space
    radius, angle = compute_polar_dynamics(kernel_path, space=20)
    jnp.save(kernel_path+'/radius_top20', radius)
    jnp.save(kernel_path+'/angle_top20', angle)
    # try:
    #     radius, angle = jnp.load(kernel_path+'/radius_top20.npy'), jnp.load(kernel_path+'/angle_top20.npy')
    # except:
    #     radius, angle = jnp.load(kernel_path+'/radius_top10.npy'), jnp.load(kernel_path+'/angle_top10.npy')
    fig = plt.figure()
    ax1 = fig.add_axes([0.1,0.1,0.8,0.8], polar=True)
    # ax1.plot(angle, radius, lw=2.5, label='std')
    for j in range(0, n-s, s):
        ax1.plot(angle[j:j+s+1], radius[j:j+s+1], color=cmap(T[j]))
    ax1.set_xlabel('radius')
    ax1.set_ylabel('angle')
    # ax1.set_xlim(0,1)
    ax1.set_title('Polar dynamics of top space')
    # plt.savefig(f'./sgd_measurements/{DATASET}/top_polar_{name_dict[i]}.pdf')
    plt.show()
    plt.close()


    # Polar phase diagram for bottom space
    radius, angle = compute_polar_dynamics(kernel_path, space=2480, top=False)
    jnp.save(kernel_path+'/radius_bottom2480', radius)
    jnp.save(kernel_path+'/angle_bottom2480', angle)
    # try:
    #     radius, angle = jnp.load(kernel_path+'/radius_top20.npy'), jnp.load(kernel_path+'/angle_top20.npy')
    # except:
    #     radius, angle = jnp.load(kernel_path+'/radius_top10.npy'), jnp.load(kernel_path+'/angle_top10.npy')
    fig = plt.figure()
    ax1 = fig.add_axes([0.1,0.1,0.8,0.8], polar=True)
    # ax1.plot(angle, radius, lw=2.5, label='std')
    for j in range(0, n-s, s):
        ax1.plot(angle[j:j+s+1], radius[j:j+s+1], color=cmap(T[j]))
    ax1.set_xlabel('radius')
    ax1.set_ylabel('angle')
    ax1.set_xlim(0, 2)
    ax1.set_title('Polar dynamics of bottom space')
    # plt.savefig(f'./sgd_measurements/{DATASET}/bottom_polar_{name_dict[i]}.pdf')
    plt.show()
    plt.close()

    # Eigenvalue mass / rank
    mass = compute_top_mass(kernel_path)
    jnp.save(kernel_path+'/mass_top10', mass['top_10'])
    jnp.save(kernel_path+'/mass_top20', mass['top_20'])
    # mass = jnp.load(kernel_path+'/mass_top20.npy')
    top = 20
    # plt.plot(mass[f'top_{top}'], lw=3, label='std train', c='navy')
    plt.plot(mass[f'top_{top}'], lw=3)
    plt.grid()
    plt.xlabel('epochs')
    plt.ylabel(r'$\frac{\sum_{i = 1}^{20} \lambda_i^2}{\sum_{i = 1}^n \lambda_i^2}$')
    plt.title(f'Evolution of mass lying at the top {top} eigenvalues')
    plt.tight_layout()
    # plt.savefig(f'./sgd_measurements/{DATASET}/top_mass_{name_dict[i]}.pdf')
    plt.show()
    plt.close()