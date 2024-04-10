from copy import deepcopy
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import gym
import pickle
import os
import sys
import random

sys.path.append("../../")
from popsan_drl.popsan_td3.replay_buffer_norm import ReplayBuffer
from popsan_drl.popsan_td3.popsan import PopSpikeActor
from popsan_drl.popsan_td3.core_cuda import MLPQFunction


class SpikeActorDeepCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 encoder_pop_dim, decoder_pop_dim, mean_range, std, spike_ts, device, use_poisson,
                 hidden_sizes=(256, 256), activation=nn.ReLU):
        """
        :param observation_space: observation space from gym
        :param action_space: action space from gym
        :param encoder_pop_dim: encoder population dimension
        :param decoder_pop_dim: decoder population dimension
        :param mean_range: mean range for encoder
        :param std: std for encoder
        :param spike_ts: spike timesteps
        :param device: device
        :param use_poisson: if true use Poisson spikes for encoder
        :param hidden_sizes: list of hidden layer sizes
        :param activation: activation function for critic network
        """
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        # build policy and value functions
        self.popsan = PopSpikeActor(obs_dim, act_dim, encoder_pop_dim, decoder_pop_dim, hidden_sizes,
                                    mean_range, std, spike_ts, act_limit, device, use_poisson)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, batch_size):
        with torch.no_grad():
            return self.popsan(obs, batch_size).to('cpu').numpy()


def spike_td3(env_fn, actor_critic=SpikeActorDeepCritic, ac_kwargs=dict(), seed=0,
              steps_per_epoch=10000, epochs=100, replay_size=int(1e6), gamma=0.99,
              polyak=0.995, popsan_lr=1e-4, q_lr=1e-3, batch_size=100, start_steps=10000,
              update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2,
              noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=1000,
              save_freq=5, norm_clip_limit=3, norm_update=50, tb_comment='', model_idx=0, use_cuda=True):
    """
    Spike Twin Delayed Deep Deterministic Policy Gradient (TD3)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``popsan`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``popsan`` module should accept batches of
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``popsan``   (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to TD3.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        popsan_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)

        target_noise (float): Stddev for smoothing noise added to target 
            policy.

        noise_clip (float): Limit for absolute value of target policy 
            smoothing noise.

        policy_delay (int): Policy will only be updated once every 
            policy_delay times for each update of the Q-networks.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        norm_clip_limit (float): Clip limit for normalize observation

        norm_update (int): Number of steps to update running mean and var in memory

        tb_comment (str): Comment for tensorboard writer

        model_idx (int): Index of training model

        use_cuda (bool): If true use cuda for computation
    """
    # Set device
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Set random seed
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    # ac_targ = deepcopy(ac)
    ac.to(device)
    # ac_targ.to(device)

    ac.popsan.load_state_dict(torch.load('params/weight.pt'))


    ############################### GN weight ##############################

    # for a in range(170):
    #    for b in range(256):
    #        ac.popsan.snn.hidden_layers[0].weight[b][a] += random.gauss(0.0,0.05)
    
    # for a in range(256):
    #    for b in range(256):
    #        ac.popsan.snn.hidden_layers[1].weight[b][a] += random.gauss(0.0,0.05)
           

    # for a in range(256):
    #    for b in range(60):
    #        ac.popsan.snn.out_pop_layer.weight[b][a] += random.gauss(0.0,0.05)

    ############################### GN weight ##############################


    ############################### 30% zero weight ##############################
    # fc11 = []
    # fc22 = []
    # fc44 = []

    # for i in range(int(0.3 * 170 * 256)):
    #     a = random.randint(0,169)
    #     b = random.randint(0,255)
    #     while [a,b] in fc11:
    #         a = random.randint(0,169)
    #         b = random.randint(0,255)
    #     ac.popsan.snn.hidden_layers[0].weight[b][a] = 0.0
    #     fc11.append([a,b])

    # for i in range(int(0.3 * 256 * 256)):
    #     a = random.randint(0,255)
    #     b = random.randint(0,255)
    #     while [a,b] in fc22:
    #         a = random.randint(0,255)
    #         b = random.randint(0,255)
    #     ac.popsan.snn.hidden_layers[1].weight[b][a] = 0.0
    #     fc22.append([a,b])

    # for i in range(int(0.3 * 60 * 256)):
    #     a = random.randint(0,255)
    #     b = random.randint(0,59)
    #     while [a,b] in fc44:
    #         a = random.randint(0,255)
    #         b = random.randint(0,59)
    #     ac.popsan.snn.out_pop_layer.weight[b][a] = 0.0
    #     fc44.append([a,b])

    ############################### 30% zero weight ##############################

    #############################  LOIHI weight  ##################################
    # w_max = max(ac.popsan.snn.hidden_layers[0].weight.max(), ac.popsan.snn.hidden_layers[1].weight.max(), ac.popsan.snn.out_pop_layer.weight.max())
    # w_min = min(ac.popsan.snn.hidden_layers[0].weight.min(), ac.popsan.snn.hidden_layers[1].weight.min(), ac.popsan.snn.out_pop_layer.weight.min())
    # b_max = max(ac.popsan.snn.hidden_layers[0].bias.max(), ac.popsan.snn.hidden_layers[1].bias.max(), ac.popsan.snn.out_pop_layer.bias.max())
    # b_min = min(ac.popsan.snn.hidden_layers[0].bias.min(), ac.popsan.snn.hidden_layers[1].bias.min(), ac.popsan.snn.out_pop_layer.bias.min())

    # w_s = 255/max(abs(w_max),abs(w_min),abs(b_max),abs(b_min))
    # ## 148.4893 ##

    # ac.popsan.snn.hidden_layers[0].weight = nn.Parameter(torch.round(torch.clamp(ac.popsan.snn.hidden_layers[0].weight * w_s, -255, 255)))
    # ac.popsan.snn.hidden_layers[1].weight = nn.Parameter(torch.round(torch.clamp(ac.popsan.snn.hidden_layers[1].weight * w_s, -255, 255)))
    # ac.popsan.snn.out_pop_layer.weight = nn.Parameter(torch.round(torch.clamp(ac.popsan.snn.out_pop_layer.weight * w_s, -255, 255)))

    # ac.popsan.snn.hidden_layers[0].bias = nn.Parameter(torch.round(torch.clamp(ac.popsan.snn.hidden_layers[0].bias * w_s, -255, 255)))
    # ac.popsan.snn.hidden_layers[1].bias = nn.Parameter(torch.round(torch.clamp(ac.popsan.snn.hidden_layers[1].bias * w_s, -255, 255)))
    # ac.popsan.snn.out_pop_layer.bias = nn.Parameter(torch.round(torch.clamp(ac.popsan.snn.out_pop_layer.bias * w_s, -255, 255)))

    #############################  LOIHI weight  ##################################



    ############################### GN weight ant ##############################

    # for a in range(1110):
    #    for b in range(256):
    #        ac.popsan.snn.hidden_layers[0].weight[b][a] += random.gauss(0.0,0.05)
    
    # for a in range(256):
    #    for b in range(256):
    #        ac.popsan.snn.hidden_layers[1].weight[b][a] += random.gauss(0.0,0.05)
           

    # for a in range(256):
    #    for b in range(80):
    #        ac.popsan.snn.out_pop_layer.weight[b][a] += random.gauss(0.0,0.05)

    ############################### GN weight ant ##############################


    ############################### 30% zero weight ant ##############################
    # fc11 = []
    # fc22 = []
    # fc44 = []

    # for i in range(int(0.3 * 1110 * 256)):
    #     a = random.randint(0,1109)
    #     b = random.randint(0,255)
    #     while [a,b] in fc11:
    #         a = random.randint(0,1109)
    #         b = random.randint(0,255)
    #     ac.popsan.snn.hidden_layers[0].weight[b][a] = 0.0
    #     fc11.append([a,b])

    # for i in range(int(0.3 * 256 * 256)):
    #     a = random.randint(0,255)
    #     b = random.randint(0,255)
    #     while [a,b] in fc22:
    #         a = random.randint(0,255)
    #         b = random.randint(0,255)
    #     ac.popsan.snn.hidden_layers[1].weight[b][a] = 0.0
    #     fc22.append([a,b])

    # for i in range(int(0.3 * 80 * 256)):
    #     a = random.randint(0,255)
    #     b = random.randint(0,79)
    #     while [a,b] in fc44:
    #         a = random.randint(0,255)
    #         b = random.randint(0,79)
    #     ac.popsan.snn.out_pop_layer.weight[b][a] = 0.0
    #     fc44.append([a,b])

    ############################### 30% zero weight ant ##############################

    #############################  LOIHI weight ant  ##################################
    # w_max = max(ac.popsan.snn.hidden_layers[0].weight.max(), ac.popsan.snn.hidden_layers[1].weight.max(), ac.popsan.snn.out_pop_layer.weight.max())
    # w_min = min(ac.popsan.snn.hidden_layers[0].weight.min(), ac.popsan.snn.hidden_layers[1].weight.min(), ac.popsan.snn.out_pop_layer.weight.min())
    # b_max = max(ac.popsan.snn.hidden_layers[0].bias.max(), ac.popsan.snn.hidden_layers[1].bias.max(), ac.popsan.snn.out_pop_layer.bias.max())
    # b_min = min(ac.popsan.snn.hidden_layers[0].bias.min(), ac.popsan.snn.hidden_layers[1].bias.min(), ac.popsan.snn.out_pop_layer.bias.min())

    # w_s = 255/max(abs(w_max),abs(w_min),abs(b_max),abs(b_min))
    # ## 148.4893 ##

    # ac.popsan.snn.hidden_layers[0].weight = nn.Parameter(torch.round(torch.clamp(ac.popsan.snn.hidden_layers[0].weight * w_s, -255, 255)))
    # ac.popsan.snn.hidden_layers[1].weight = nn.Parameter(torch.round(torch.clamp(ac.popsan.snn.hidden_layers[1].weight * w_s, -255, 255)))
    # ac.popsan.snn.out_pop_layer.weight = nn.Parameter(torch.round(torch.clamp(ac.popsan.snn.out_pop_layer.weight * w_s, -255, 255)))

    # ac.popsan.snn.hidden_layers[0].bias = nn.Parameter(torch.round(torch.clamp(ac.popsan.snn.hidden_layers[0].bias * w_s, -255, 255)))
    # ac.popsan.snn.hidden_layers[1].bias = nn.Parameter(torch.round(torch.clamp(ac.popsan.snn.hidden_layers[1].bias * w_s, -255, 255)))
    # ac.popsan.snn.out_pop_layer.bias = nn.Parameter(torch.round(torch.clamp(ac.popsan.snn.out_pop_layer.bias * w_s, -255, 255)))

    #############################  LOIHI weight ant  ##################################

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size,
                                 clip_limit=norm_clip_limit, norm_update_every=norm_update)


    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32, device=device), 1)
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        ###
        # compuate the return mean test reward
        ###
        test_reward_sum = 0
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            
            # pose noise hc
            #i = random.randint(0,7)

            # velocity noise hc
            # i = random.randint(8,16)

            # pose noise ant
            #i = random.randint(0,12)

            # velocity noise ant
            # i = random.randint(13,26)
            while not(d or (ep_len == max_ep_len)):

                
                # GN 
                # for i in range(len(o)):
                #     o[i] += random.gauss(0.0,0.1)

                # pose noise                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
                #o[i] = random.gauss(0.0,0.1)

                # velocity noise
                # o[i] = random.gauss(0.0,10.0)


                # test_env.render()

                o, r, d, _ = test_env.step(get_action(replay_buffer.normalize_obs(o), 0))
                ep_ret += r
                ep_len += 1
            test_reward_sum += ep_ret
        return test_reward_sum / num_test_episodes

   
    total_steps = steps_per_epoch * epochs
   
    for t in range(total_steps):
        
        test_mean_reward = test_agent()
        print("Model: ", model_idx, " Steps: ", t + 1, " Mean Reward: ", test_mean_reward)

   

if __name__ == '__main__':
    import math
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Ant-v3')
    parser.add_argument('--encoder_pop_dim', type=int, default=10)
    parser.add_argument('--decoder_pop_dim', type=int, default=10)
    parser.add_argument('--encoder_var', type=float, default=0.15)
    parser.add_argument('--start_model_idx', type=int, default=0)
    parser.add_argument('--num_model', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    START_MODEL = args.start_model_idx
    NUM_MODEL = args.num_model
    USE_POISSON = False
    if args.env == 'Hopper-v3' or args.env == 'Walker2d-v3':
        USE_POISSON = True
    AC_KWARGS = dict(hidden_sizes=[256, 256],
                     encoder_pop_dim=args.encoder_pop_dim,
                     decoder_pop_dim=args.decoder_pop_dim,
                     mean_range=(-3, 3),
                     std=math.sqrt(args.encoder_var),
                     spike_ts=5,
                     device=torch.device('cuda'),
                     use_poisson=USE_POISSON)
    COMMENT = "td3-popsan-" + args.env + "-encoder-dim-" + str(AC_KWARGS['encoder_pop_dim']) + \
              "-decoder-dim-" + str(AC_KWARGS['decoder_pop_dim'])
    for num in range(START_MODEL, START_MODEL + NUM_MODEL):
        seed = num * 10
        spike_td3(lambda : gym.make(args.env), actor_critic=SpikeActorDeepCritic, ac_kwargs=AC_KWARGS,
                  popsan_lr=1e-4, gamma=0.99, seed=seed, epochs=args.epochs,
                  norm_clip_limit=3.0, tb_comment=COMMENT, model_idx=num)
