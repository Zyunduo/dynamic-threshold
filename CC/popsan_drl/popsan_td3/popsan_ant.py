import torch
import torch.nn as nn
import numpy as np
import random


"""
Parameters for SNN
"""

ENCODER_REGULAR_VTH = 0.999
NEURON_VTH = 0.5
NEURON_CDECAY = 1 / 2
NEURON_VDECAY = 3 / 4
SPIKE_PSEUDO_GRAD_WINDOW = 0.5

NEURON_VTH_1 = torch.full((1,256),0.5)

NEURON_VTH_1_temporal = torch.full((1,256),0.5)

NEURON_VTH_2 = torch.full((1,256),0.5)

NEURON_VTH_2_temporal = torch.full((1,256),0.5)

NEURON_VTH_3 = torch.full((1,80),0.5)

NEURON_VTH_3_temporal = torch.full((1,80),0.5)

class PseudoEncoderSpikeRegular(torch.autograd.Function):
    """ Pseudo-gradient function for spike - Regular Spike for encoder """
    @staticmethod
    def forward(ctx, input):
        return input.gt(ENCODER_REGULAR_VTH).float()
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class PseudoEncoderSpikePoisson(torch.autograd.Function):
    """ Pseudo-gradient function for spike - Poisson Spike for encoder """
    @staticmethod
    def forward(ctx, input):
        return torch.bernoulli(input).float()
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class PopSpikeEncoderRegularSpike(nn.Module):
    """ Learnable Population Coding Spike Encoder with Regular Spike Trains """
    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std, device):
        """
        :param obs_dim: observation dimension
        :param pop_dim: population dimension
        :param spike_ts: spike timesteps
        :param mean_range: mean range
        :param std: std
        :param device: device
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.pop_dim = pop_dim
        self.encoder_neuron_num = obs_dim * pop_dim
        self.spike_ts = spike_ts
        self.device = device
        self.pseudo_spike = PseudoEncoderSpikeRegular.apply
        # Compute evenly distributed mean and variance
        tmp_mean = torch.zeros(1, obs_dim, pop_dim)
        delta_mean = (mean_range[1] - mean_range[0]) / (pop_dim - 1)
        for num in range(pop_dim):
            tmp_mean[0, :, num] = mean_range[0] + delta_mean * num
        tmp_std = torch.zeros(1, obs_dim, pop_dim) + std
        self.mean = nn.Parameter(tmp_mean)
        self.std = nn.Parameter(tmp_std)

    def forward(self, obs, batch_size):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: pop_spikes
        """
        obs = obs.view(-1, self.obs_dim, 1)
        # Receptive Field of encoder population has Gaussian Shape
        pop_act = torch.exp(-(1. / 2.) * (obs - self.mean).pow(2) / self.std.pow(2)).view(-1, self.encoder_neuron_num)
        pop_volt = torch.zeros(batch_size, self.encoder_neuron_num, device=self.device)
        pop_spikes = torch.zeros(batch_size, self.encoder_neuron_num, self.spike_ts, device=self.device)
        # Generate Regular Spike Trains
        for step in range(self.spike_ts):
            pop_volt = pop_volt + pop_act
            pop_spikes[:, :, step] = self.pseudo_spike(pop_volt)
            pop_volt = pop_volt - pop_spikes[:, :, step] * ENCODER_REGULAR_VTH
        return pop_spikes


class PopSpikeEncoderPoissonSpike(PopSpikeEncoderRegularSpike):
    """ Learnable Population Coding Spike Encoder with Poisson Spike Trains """
    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std, device):
        """
        :param obs_dim: observation dimension
        :param pop_dim: population dimension
        :param spike_ts: spike timesteps
        :param mean_range: mean range
        :param std: std
        :param device: device
        """
        super().__init__(obs_dim, pop_dim, spike_ts, mean_range, std, device)
        self.pseudo_spike = PseudoEncoderSpikePoisson.apply

    def forward(self, obs, batch_size):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: pop_spikes
        """
        obs = obs.view(-1, self.obs_dim, 1)
        # Receptive Field of encoder population has Gaussian Shape
        pop_act = torch.exp(-(1. / 2.) * (obs - self.mean).pow(2) / self.std.pow(2)).view(-1, self.encoder_neuron_num)
        pop_spikes = torch.zeros(batch_size, self.encoder_neuron_num, self.spike_ts, device=self.device)
        # Generate Poisson Spike Trains
        for step in range(self.spike_ts):
            pop_spikes[:, :, step] = self.pseudo_spike(pop_act)
        return pop_spikes


class PopSpikeDecoder(nn.Module):
    """ Population Coding Spike Decoder """
    def __init__(self, act_dim, pop_dim, output_activation=nn.Tanh):
        """
        :param act_dim: action dimension
        :param pop_dim:  population dimension
        :param output_activation: activation function added on output
        """
        super().__init__()
        self.act_dim = act_dim
        self.pop_dim = pop_dim
        self.decoder = nn.Conv1d(act_dim, act_dim, pop_dim, groups=act_dim)
        self.output_activation = output_activation()

    def forward(self, pop_act):
        """
        :param pop_act: output population activity
        :return: raw_act
        """
        pop_act = pop_act.view(-1, self.act_dim, self.pop_dim)
        raw_act = self.output_activation(self.decoder(pop_act).view(-1, self.act_dim))
        return raw_act


class PseudoSpikeRect_1(torch.autograd.Function):
    """ Pseudo-gradient function for spike - Derivative of Rect Function """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(NEURON_VTH_1.cuda()).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        spike_pseudo_grad = (abs(input - NEURON_VTH_1.cuda()) < SPIKE_PSEUDO_GRAD_WINDOW)
        return grad_input * spike_pseudo_grad.float()

class PseudoSpikeRect_2(torch.autograd.Function):
    """ Pseudo-gradient function for spike - Derivative of Rect Function """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(NEURON_VTH_2.cuda()).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        spike_pseudo_grad = (abs(input - NEURON_VTH_2.cuda()) < SPIKE_PSEUDO_GRAD_WINDOW)
        return grad_input * spike_pseudo_grad.float()

class PseudoSpikeRect_3(torch.autograd.Function):
    """ Pseudo-gradient function for spike - Derivative of Rect Function """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(NEURON_VTH_3.cuda()).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        spike_pseudo_grad = (abs(input - NEURON_VTH_3.cuda()) < SPIKE_PSEUDO_GRAD_WINDOW)
        return grad_input * spike_pseudo_grad.float()

class PseudoSpikeRect(torch.autograd.Function):
    """ Pseudo-gradient function for spike - Derivative of Rect Function """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(NEURON_VTH).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        spike_pseudo_grad = (abs(input - NEURON_VTH) < SPIKE_PSEUDO_GRAD_WINDOW)
        return grad_input * spike_pseudo_grad.float()


class SpikeMLP(nn.Module):
    """ Spike MLP with Input and Output population neurons """
    def __init__(self, in_pop_dim, out_pop_dim, hidden_sizes, spike_ts, device):
        """
        :param in_pop_dim: input population dimension
        :param out_pop_dim: output population dimension
        :param hidden_sizes: list of hidden layer sizes
        :param spike_ts: spike timesteps
        :param device: device
        """
        super().__init__()
        self.in_pop_dim = in_pop_dim
        self.out_pop_dim = out_pop_dim
        self.hidden_sizes = hidden_sizes
        self.hidden_num = len(hidden_sizes)
        self.spike_ts = spike_ts
        self.device = device
        self.pseudo_spike_1 = PseudoSpikeRect_1.apply
        self.pseudo_spike_2 = PseudoSpikeRect_2.apply
        self.pseudo_spike_3 = PseudoSpikeRect_3.apply
        self.pseudo_spike = PseudoSpikeRect.apply
        # Define Layers (Hidden Layers + Output Population)
        self.hidden_layers = nn.ModuleList([nn.Linear(in_pop_dim, hidden_sizes[0])])
        if self.hidden_num > 1:
            for layer in range(1, self.hidden_num):
                self.hidden_layers.extend([nn.Linear(hidden_sizes[layer-1], hidden_sizes[layer])])
        self.out_pop_layer = nn.Linear(hidden_sizes[-1], out_pop_dim)

    def neuron_model_dth(self, syn_func, pre_layer_output, current, volt_, spike):
        """
        LIF Neuron Model
        :param syn_func: synaptic function
        :param pre_layer_output: output from pre-synaptic layer
        :param current: current of last step
        :param volt: voltage of last step
        :param spike: spike of last step
        :return: current, volt, spike
        """
        global NEURON_VTH_1
        global NEURON_VTH_1_temporal

        current = current * NEURON_CDECAY + syn_func(pre_layer_output)
        volt = volt_ * NEURON_VDECAY * (1. - spike) + current
        a = np.exp(- abs(NEURON_VTH_1.mean().cuda().data.cpu().numpy()))
        NEURON_VTH_1_energy = np.exp((volt_.cuda().data.cpu().numpy() - volt.cuda().data.cpu().numpy())/3) - 1
        # print(torch.from_numpy(NEURON_VTH_1_energy).shape)
        # print(NEURON_VTH_1_energy.shape)
        if NEURON_VTH_1_energy.shape[0] == 1 and NEURON_VTH_1_temporal.shape[0] == 100:
            NEURON_VTH_1_temporal = torch.full((1,256),0.5)
        NEURON_VTH_1 = 0.5 * NEURON_VTH_1_temporal.cuda() + 0.5 * torch.from_numpy(NEURON_VTH_1_energy).cuda()
        spike = self.pseudo_spike_1(volt)
        return current, volt, spike

    def neuron_model_dth2(self, syn_func, pre_layer_output, current, volt_, spike):
        """
        LIF Neuron Model
        :param syn_func: synaptic function
        :param pre_layer_output: output from pre-synaptic layer
        :param current: current of last step
        :param volt: voltage of last step
        :param spike: spike of last step
        :return: current, volt, spike
        """
        global NEURON_VTH_2
        global NEURON_VTH_2_temporal

        current = current * NEURON_CDECAY + syn_func(pre_layer_output)
        volt = volt_ * NEURON_VDECAY * (1. - spike) + current
        a = np.exp(- abs(NEURON_VTH_2.mean().cuda().data.cpu().numpy()))
        NEURON_VTH_2_energy = np.exp((volt_.cuda().data.cpu().numpy() - volt.cuda().data.cpu().numpy())/3) - 1
        # print(torch.from_numpy(NEURON_VTH_1_energy).shape)
        # print(NEURON_VTH_1_energy.shape)
        if NEURON_VTH_2_energy.shape[0] == 1 and NEURON_VTH_2_temporal.shape[0] == 100:
            NEURON_VTH_2_temporal = torch.full((1,256),0.5)
        NEURON_VTH_2 = 0.5 * NEURON_VTH_2_temporal.cuda() + 0.5 * torch.from_numpy(NEURON_VTH_2_energy).cuda()
        spike = self.pseudo_spike_2(volt)
        return current, volt, spike

    def neuron_model_dth3(self, syn_func, pre_layer_output, current, volt_, spike):
        """
        LIF Neuron Model
        :param syn_func: synaptic function
        :param pre_layer_output: output from pre-synaptic layer
        :param current: current of last step
        :param volt: voltage of last step
        :param spike: spike of last step
        :return: current, volt, spike
        """
        global NEURON_VTH_3
        global NEURON_VTH_3_temporal

        current = current * NEURON_CDECAY + syn_func(pre_layer_output)
        volt = volt_ * NEURON_VDECAY * (1. - spike) + current
        a = np.exp(- abs(NEURON_VTH_3.mean().cuda().data.cpu().numpy()))
        NEURON_VTH_3_energy = np.exp((volt_.cuda().data.cpu().numpy() - volt.cuda().data.cpu().numpy())/3) - 1
        # print(NEURON_VTH_3_energy.shape)
        # print(torch.from_numpy(NEURON_VTH_1_energy).shape)
        # print(NEURON_VTH_1_energy.shape)
        if NEURON_VTH_3_energy.shape[0] == 1 and NEURON_VTH_3_temporal.shape[0] == 100:
            NEURON_VTH_3_temporal = torch.full((1,80),0.5)
        # print(NEURON_VTH_3_temporal.shape)
        NEURON_VTH_3 = 0.5 * NEURON_VTH_3_temporal.cuda() + 0.5 * torch.from_numpy(NEURON_VTH_3_energy).cuda()
        spike = self.pseudo_spike_3(volt)
        return current, volt, spike


    def neuron_model(self, syn_func, pre_layer_output, current, volt, spike):
        """
        LIF Neuron Model
        :param syn_func: synaptic function
        :param pre_layer_output: output from pre-synaptic layer
        :param current: current of last step
        :param volt: voltage of last step
        :param spike: spike of last step
        :return: current, volt, spike
        """
        current = current * NEURON_CDECAY + syn_func(pre_layer_output)
        volt = volt * NEURON_VDECAY * (1. - spike) + current
        spike = self.pseudo_spike(volt)
        return current, volt, spike
    

    def forward(self, in_pop_spikes, batch_size):
        """
        :param in_pop_spikes: input population spikes
        :param batch_size: batch size
        :return: out_pop_act
        """
        # Define LIF Neuron states: Current, Voltage, and Spike
        global NEURON_VTH_1
        global NEURON_VTH_2
        global NEURON_VTH_3
        global NEURON_VTH_1_temporal
        global NEURON_VTH_2_temporal
        global NEURON_VTH_3_temporal
        hidden_states = []
        for layer in range(self.hidden_num):
            hidden_states.append([torch.zeros(batch_size, self.hidden_sizes[layer], device=self.device)
                                  for _ in range(3)])
        out_pop_states = [torch.zeros(batch_size, self.out_pop_dim, device=self.device)
                          for _ in range(3)]
        out_pop_act = torch.zeros(batch_size, self.out_pop_dim, device=self.device)
        # Start Spike Timestep Iteration
        for step in range(self.spike_ts):
            # print(self.spike_ts)
            in_pop_spike_t = in_pop_spikes[:, :, step]
            hidden_states[0][0], hidden_states[0][1], hidden_states[0][2] = self.neuron_model_dth(
                self.hidden_layers[0], in_pop_spike_t,
                hidden_states[0][0], hidden_states[0][1], hidden_states[0][2]
            )
            # print('*************')
            # print(hidden_states[0][1].shape)
            # if hidden_states[0][1].shape[0] == 1:
            V_m1 = hidden_states[0][1].mean() - 0.2 * (hidden_states[0][1].max()-hidden_states[0][1].min())
            V_theta1 = NEURON_VTH_1.mean() - 0.2 * (NEURON_VTH_1.max()-NEURON_VTH_1.min())
            NEURON_VTH_1_temporal = 0.01 * (hidden_states[0][1] - V_m1) + V_theta1 + torch.log(1 + torch.exp((hidden_states[0][1] - V_m1)/6))
            if self.hidden_num > 1:
                for layer in range(1, self.hidden_num):
                    hidden_states[layer][0], hidden_states[layer][1], hidden_states[layer][2] = self.neuron_model_dth2(
                        self.hidden_layers[layer], hidden_states[layer-1][2],
                        hidden_states[layer][0], hidden_states[layer][1], hidden_states[layer][2]
                    )
                    V_m2 = hidden_states[layer][1].mean() - 0.2 * (hidden_states[layer][1].max()-hidden_states[layer][1].min())
                    V_theta2 = NEURON_VTH_2.mean() - 0.2 * (NEURON_VTH_2.max()-NEURON_VTH_2.min())
                    NEURON_VTH_2_temporal = 0.01 * (hidden_states[layer][1] - V_m2) + V_theta2 + torch.log(1 + torch.exp((hidden_states[layer][1] - V_m2)/6))
            
            out_pop_states[0], out_pop_states[1], out_pop_states[2] = self.neuron_model_dth3(
                self.out_pop_layer, hidden_states[-1][2],
                out_pop_states[0], out_pop_states[1], out_pop_states[2]
            )
            V_m3 = out_pop_states[1].mean() - 0.2 * (out_pop_states[1].max()-out_pop_states[1].min())
            V_theta3 = NEURON_VTH_3.mean() - 0.2 * (NEURON_VTH_3.max()-NEURON_VTH_3.min())
            NEURON_VTH_3_temporal = 0.01 * (out_pop_states[1] - V_m3) + V_theta3 + torch.log(1 + torch.exp((out_pop_states[1] - V_m3)/6))
            
            # print(out_pop_states[2].shsape)
            out_pop_act += out_pop_states[2]
        out_pop_act = out_pop_act / self.spike_ts
        return out_pop_act


class PopSpikeActor(nn.Module):
    """ Population Coding Spike Actor with Fix Encoder """
    def __init__(self, obs_dim, act_dim, en_pop_dim, de_pop_dim, hidden_sizes,
                 mean_range, std, spike_ts, act_limit, device, use_poisson):
        """
        :param obs_dim: observation dimension
        :param act_dim: action dimension
        :param en_pop_dim: encoder population dimension
        :param de_pop_dim: decoder population dimension
        :param hidden_sizes: list of hidden layer sizes
        :param mean_range: mean range for encoder
        :param std: std for encoder
        :param spike_ts: spike timesteps
        :param act_limit: action limit
        :param device: device
        :param use_poisson: if true use Poisson spikes for encoder
        """
        super().__init__()
        self.act_limit = act_limit
        if use_poisson:
            self.encoder = PopSpikeEncoderPoissonSpike(obs_dim, en_pop_dim, spike_ts, mean_range, std, device)
        else:
            self.encoder = PopSpikeEncoderRegularSpike(obs_dim, en_pop_dim, spike_ts, mean_range, std, device)
        self.snn = SpikeMLP(obs_dim*en_pop_dim, act_dim*de_pop_dim, hidden_sizes, spike_ts, device)
        self.decoder = PopSpikeDecoder(act_dim, de_pop_dim)

    def forward(self, obs, batch_size):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: action scale with action limit
        """
        in_pop_spikes = self.encoder(obs, batch_size)
        out_pop_activity = self.snn(in_pop_spikes, batch_size)
        action = self.act_limit * self.decoder(out_pop_activity)
        return action
